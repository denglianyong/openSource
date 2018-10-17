//  Copyright 2018 Istio Authors
//
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.

package echo

import (
	"fmt"
	"net"
	"net/http"

	multierror "github.com/hashicorp/go-multierror"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials"

	"istio.io/istio/pilot/pkg/model"
	"istio.io/istio/pkg/log"
	"istio.io/istio/pkg/test/protocol"
	"istio.io/istio/pkg/test/service/echo/proto"
)

// Application is a simple application than processes echo requests via various transports.
type Application struct {
	// Ports are the ports that the application should listen on. If any port number is 0, an available port will be selected
	// when the application is started.
	Ports model.PortList
	// TLSCert defines the server-side TLS cert to use with GRPC.
	TLSCert string
	// TLSKey defines the server-side TLS key to use with GRPC.
	TLSCKey string
	// Version string
	Version string
	// Client client for calling out to other services
	Client protocol.Client

	servers []serverInterface
}

// GetPorts returns the ports for this application.
func (a *Application) GetPorts() model.PortList {
	return a.Ports
}

// Start the application.
func (a *Application) Start() (err error) {
	defer func() {
		if err != nil {
			a.Close()
		}
	}()

	if err = a.validate(); err != nil {
		return err
	}

	a.servers = make([]serverInterface, len(a.Ports))
	for i, p := range a.Ports {
		handler := &handler{
			version: a.Version,
			caFile:  a.TLSCert,
			client:  a.Client,
		}
		switch p.Protocol {
		case model.ProtocolTCP:
			fallthrough
		case model.ProtocolHTTP:
			fallthrough
		case model.ProtocolHTTPS:
			a.servers[i] = &httpServer{
				port: p,
				h:    handler,
			}
		case model.ProtocolHTTP2:
			fallthrough
		case model.ProtocolGRPC:
			a.servers[i] = &grpcServer{
				port:    p,
				h:       handler,
				tlsCert: a.TLSCert,
				tlsCKey: a.TLSCKey,
			}
		default:
			return fmt.Errorf("unsupported protocol: %s", p.Protocol)
		}
	}

	// Start the servers, updating port numbers as necessary.
	for _, s := range a.servers {
		if err := s.start(); err != nil {
			return err
		}
	}
	return nil
}

// Close stops this application
func (a *Application) Close() (err error) {
	for i, s := range a.servers {
		if s != nil {
			err = multierror.Append(err, s.stop())
			a.servers[i] = nil
		}
	}
	return
}

func (a *Application) validate() error {
	for _, port := range a.Ports {
		switch port.Protocol {
		case model.ProtocolTCP:
		case model.ProtocolHTTP:
		case model.ProtocolHTTPS:
		case model.ProtocolHTTP2:
		case model.ProtocolGRPC:
		default:
			return fmt.Errorf("protocol %v not currently supported", port.Protocol)
		}
	}
	return nil
}

type serverInterface interface {
	start() error
	stop() error
}

type httpServer struct {
	server *http.Server
	port   *model.Port
	h      *handler
}

func (s *httpServer) start() error {
	// Listen on the given port and update the port if it changed from what was passed in.
	listener, p, err := listenOnPort(s.port.Port)
	if err != nil {
		return err
	}
	// Store the actual listening port back to the argument.
	s.port.Port = p
	s.h.port = p
	fmt.Printf("Listening HTTP/1.1 on %v\n", p)

	s.server = &http.Server{
		Addr:    fmt.Sprintf(":%d", p),
		Handler: s.h,
	}

	// Start serving HTTP traffic.
	go s.server.Serve(listener)
	return nil
}

func (s *httpServer) stop() error {
	return s.server.Close()
}

type grpcServer struct {
	tlsCert string
	tlsCKey string
	version string
	port    *model.Port
	h       *handler

	server *grpc.Server
}

func (s *grpcServer) start() error {
	// Listen on the given port and update the port if it changed from what was passed in.
	listener, p, err := listenOnPort(s.port.Port)
	if err != nil {
		return err
	}
	// Store the actual listening port back to the argument.
	s.port.Port = p
	s.h.port = p
	fmt.Printf("Listening GRPC on %v\n", p)

	if s.tlsCert != "" && s.tlsCKey != "" {
		// Create the TLS credentials
		creds, errCreds := credentials.NewServerTLSFromFile(s.tlsCert, s.tlsCKey)
		if errCreds != nil {
			log.Errorf("could not load TLS keys: %s", errCreds)
		}
		s.server = grpc.NewServer(grpc.Creds(creds))
	} else {
		s.server = grpc.NewServer()
	}
	proto.RegisterEchoTestServiceServer(s.server, s.h)

	// Start serving GRPC traffic.
	go s.server.Serve(listener)
	return nil
}

func (s *grpcServer) stop() error {
	s.server.Stop()
	return nil
}

func listenOnPort(port int) (net.Listener, int, error) {
	ln, err := net.Listen("tcp", fmt.Sprintf(":%d", port))
	if err != nil {
		return nil, 0, err
	}

	port = ln.Addr().(*net.TCPAddr).Port
	return ln, port, nil
}

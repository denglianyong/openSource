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

package server

import (
	"fmt"
	"net"
	"strings"
	"time"

	"google.golang.org/grpc"

	"istio.io/istio/pkg/mcp/creds"

	mcp "istio.io/api/mcp/v1alpha1"
	"istio.io/istio/galley/pkg/kube/source"
	"istio.io/istio/galley/pkg/metadata"

	"istio.io/istio/galley/pkg/kube"
	"istio.io/istio/galley/pkg/runtime"
	"istio.io/istio/pkg/ctrlz"
	"istio.io/istio/pkg/log"
	"istio.io/istio/pkg/mcp/server"
	"istio.io/istio/pkg/mcp/snapshot"
	"istio.io/istio/pkg/probe"
)

// Server is the main entry point into the Galley code.
type Server struct {
	shutdown chan error

	grpcServer *grpc.Server
	processor  *runtime.Processor
	mcp        *server.Server
	listener   net.Listener
	controlZ   *ctrlz.Server
	stopCh     chan struct{}

	// probes
	livenessProbe  probe.Controller
	readinessProbe probe.Controller
	*probe.Probe  // 继承Probe
}

type patchTable struct {
	logConfigure          func(*log.Options) error
	newKubeFromConfigFile func(string) (kube.Interfaces, error)
	newSource             func(kube.Interfaces, time.Duration) (runtime.Source, error)
	netListen             func(network, address string) (net.Listener, error)
}

func defaultPatchTable() patchTable {
	return patchTable{
		logConfigure:          log.Configure,
		newKubeFromConfigFile: kube.NewKubeFromConfigFile,//F:\go-source\istio-master\galley\pkg\kube\interfaces.go
		newSource:             source.New,//F:\go-source\istio-master\galley\pkg\kube\source\source.go
		netListen:             net.Listen,
	}
}

// New returns a new instance of a Server.
func New(a *Args) (*Server, error) {
	return newServer(a, defaultPatchTable())
}

func newServer(a *Args, p patchTable) (*Server, error) {
	s := &Server{}

	if err := p.logConfigure(a.LoggingOptions); err != nil {
		return nil, err
	}
//F:\go-source\istio-master\galley\pkg\kube\interfaces.go  访问k8s
	k, err := p.newKubeFromConfigFile(a.KubeConfig)
	if err != nil {
		return nil, err
	}
//F:\go-source\istio-master\galley\pkg\kube\source\source.go  的New
// 从k8s上获取event
	src, err := p.newSource(k, a.ResyncPeriod)
	if err != nil {
		return nil, err
	}
	//F:\go-source\istio-master\pkg\mcp\snapshot\snapshot.go
	//就是一个Cache
	distributor := snapshot.New()
	// 根据k8s  上 event 更新配置，状态，做snapshot 
	//F:\go-source\istio-master\galley\pkg\runtime\processor.go
	s.processor = runtime.NewProcessor(src, distributor)

	var grpcOptions []grpc.ServerOption
	grpcOptions = append(grpcOptions, grpc.MaxConcurrentStreams(uint32(a.MaxConcurrentStreams)))
	grpcOptions = append(grpcOptions, grpc.MaxRecvMsgSize(int(a.MaxReceivedMessageSize)))

	s.stopCh = make(chan struct{})
	//检查权限
	var checker *server.ListAuthChecker
	if !a.Insecure {
// 访问控制列表
		checker, err = watchAccessList(s.stopCh, a.AccessListFile)
		if err != nil {
			return nil, err
		}
//F:\go-source\istio-master\pkg\mcp\creds\watcher.go  CA证书
		watcher, err := creds.WatchFiles(s.stopCh, a.CredentialOptions)
		if err != nil {
			return nil, err
		}
		//F:\go-source\istio-master\pkg\mcp\creds\create.go   MCP server
		credentials := creds.CreateForServer(watcher)

		grpcOptions = append(grpcOptions, grpc.Creds(credentials))
	}
	grpc.EnableTracing = a.EnableGRPCTracing

	s.grpcServer = grpc.NewServer(grpcOptions...)
	// MCP server 分发配置
// mesh  config protocol   F:\go-source\istio-master\pkg\mcp\server\server.go
	s.mcp = server.New(distributor, metadata.Types.TypeURLs(), checker)

	// get the network stuff setup
	network := "tcp"
	var address string
	idx := strings.Index(a.APIAddress, "://")
	if idx < 0 {
		address = a.APIAddress
	} else {
		network = a.APIAddress[:idx]
		address = a.APIAddress[idx+3:]
	}

	if s.listener, err = p.netListen(network, address); err != nil {
		_ = s.Close()
		return nil, fmt.Errorf("unable to listen: %v", err)
	}
//  grpc 与 MCP server  整合在一起 
	mcp.RegisterAggregatedMeshConfigServiceServer(s.grpcServer, s.mcp)
// 健康检查--检查负载均衡池的其他服务 F:\go-source\istio-master\pkg\probe\probe.go
	s.Probe = probe.NewProbe()

	if a.LivenessProbeOptions.IsValid() {
		//F:\go-source\istio-master\pkg\probe\controller.go
		s.livenessProbe = probe.NewFileController(a.LivenessProbeOptions)
		// s继承probe的RegisterProbe
		s.RegisterProbe(s.livenessProbe, "server")
		// 开始探测，到齐后生成文件
		s.livenessProbe.Start()
	}

	if a.ReadinessProbeOptions.IsValid() {
		s.readinessProbe = probe.NewFileController(a.ReadinessProbeOptions)
		s.readinessProbe.Start()
	}
// 自服务   暴露MCP 状态 F:\go-source\istio-master\pkg\ctrlz\ctrlz.go
	s.controlZ, _ = ctrlz.Run(a.IntrospectionOptions, nil)

	return s, nil
}

// Run enables Galley to start receiving gRPC requests on its main API port.
func (s *Server) Run() {
	s.shutdown = make(chan error, 1)
	s.SetAvailable(nil)
	go func() {
		// 更新配置 	//F:\go-source\istio-master\galley\pkg\runtime\processor.go
		err := s.processor.Start()
		if err != nil {
			s.shutdown <- err
			return
		}

		// start serving
		err = s.grpcServer.Serve(s.listener)
		// notify closer we're done
		s.shutdown <- err
	}()
}

// Wait waits for the server to exit.
func (s *Server) Wait() error {
	if s.shutdown == nil {
		return fmt.Errorf("server not running")
	}

	err := <-s.shutdown
	s.shutdown = nil
	return err
}

// Close cleans up resources used by the server.
func (s *Server) Close() error {
	if s.stopCh != nil {
		close(s.stopCh)
		s.stopCh = nil
	}

	if s.shutdown != nil {
		s.grpcServer.GracefulStop()
		_ = s.Wait()
	}

	if s.controlZ != nil {
		s.controlZ.Close()
	}

	if s.processor != nil {
		s.processor.Stop()
	}

	if s.listener != nil {
		_ = s.listener.Close()
	}

	if s.livenessProbe != nil {
		_ = s.livenessProbe.Close()
	}

	if s.readinessProbe != nil {
		_ = s.readinessProbe.Close()
	}

	// final attempt to purge buffered logs
	_ = log.Sync()

	return nil
}

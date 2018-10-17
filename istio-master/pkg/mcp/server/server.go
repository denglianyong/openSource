// Copyright 2018 Istio Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package server

import (
	"fmt"
	"io"
	"strconv"
	"strings"
	"sync/atomic"

	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/credentials"
	"google.golang.org/grpc/peer"
	"google.golang.org/grpc/status"

	mcp "istio.io/api/mcp/v1alpha1"
	"istio.io/istio/pkg/log"
)

var (
	scope = log.RegisterScope("mcp", "mcp debugging", 0)
)

// WatchResponse contains a versioned collection of pre-serialized resources.
type WatchResponse struct {
	TypeURL string

	// Version of the resources in the response for the given
	// type. The client responses with this version in subsequent
	// requests as an acknowledgment.
	Version string

	// Enveloped resources to be included in the response.
	Envelopes []*mcp.Envelope
}

// CancelWatchFunc allows the consumer to cancel a previous watch,
// terminating the watch for the request.
type CancelWatchFunc func()

// Watcher requests watches for configuration resources by node, last
// applied version, and type. The watch should send the responses when
// they are ready. The watch can be canceled by the consumer.
type Watcher interface {
	// Watch returns a new open watch for a non-empty request.
	//
	// Immediate responses should be returned to the caller along
	// with an optional cancel function. Asynchronous responses should
	// be delivered through the write-only WatchResponse channel. If the
	// channel is closed prior to cancellation of the watch, an
	// unrecoverable error has occurred in the producer, and the consumer
	// should close the corresponding stream.
	//
	// Cancel is an optional function to release resources in the
	// producer. It can be called idempotently to cancel and release resources.
	Watch(*mcp.MeshConfigRequest, chan<- *WatchResponse) (*WatchResponse, CancelWatchFunc)
}

var _ mcp.AggregatedMeshConfigServiceServer = &Server{}

// Server implements the Mesh Configuration Protocol (MCP) gRPC server.
type Server struct {
	watcher        Watcher
	supportedTypes []string
	nextStreamID   int64
	authCheck      AuthChecker
}

// AuthChecker is used to check the transport auth info that is associated with each stream. If the function
// returns nil, then the connection will be allowed. If the function returns an error, then it will be
// percolated up to the gRPC stack.
//
// Note that it is possible that this method can be called with nil authInfo. This can happen either if there
// is no peer info, or if the underlying gRPC stream is insecure. The implementations should be resilient in
// this case and apply appropriate policy.
type AuthChecker interface {
	Check(authInfo credentials.AuthInfo) error
}

// watch maintains local state of the most recent watch per-type.
type watch struct {
	cancel func()
	nonce  string
}

// connection maintains per-stream connection state for a
// client. Access to the stream and watch state is serialized
// through request and response channels.
type connection struct {
	peerAddr string
	stream   mcp.AggregatedMeshConfigService_StreamAggregatedResourcesServer
	id       int64

	// unique nonce generator for req-resp pairs per xDS stream; the server
	// ignores stale nonces. nonce is only modified within send() function.
	streamNonce int64

	requestC  chan *mcp.MeshConfigRequest // a channel for receiving incoming requests
	reqError  error                       // holds error if request channel is closed
	responseC chan *WatchResponse         // channel of pushes responses
	watches   map[string]*watch           // per-type watches
	watcher   Watcher
}

// New creates a new gRPC server that implements the Mesh Configuration Protocol (MCP). A nil authCheck
// implies all incoming connections should be allowed.
func New(watcher Watcher, supportedTypes []string, authChecker AuthChecker) *Server {
	return &Server{
		watcher:        watcher,//F:\go-source\istio-master\pkg\mcp\snapshot\snapshot.go 
		supportedTypes: supportedTypes,
		authCheck:      authChecker,
	}
}

func (s *Server) newConnection(stream mcp.AggregatedMeshConfigService_StreamAggregatedResourcesServer) (*connection, error) {
	peerAddr := "0.0.0.0"

	peerInfo, ok := peer.FromContext(stream.Context())
	if ok {
		peerAddr = peerInfo.Addr.String()
	} else {
		log.Warnf("No peer info found on the incoming stream.")
		peerInfo = nil
	}

	var authInfo credentials.AuthInfo
	if peerInfo != nil {
		authInfo = peerInfo.AuthInfo
	}

	if s.authCheck != nil {
		if err := s.authCheck.Check(authInfo); err != nil {
			log.Infof("newConnection: auth check handler returned error: %v", err)
			return nil, status.Errorf(codes.Unauthenticated, "Authentication failure: %v", err)
		}
	}

	con := &connection{
		stream:    stream,
		peerAddr:  peerAddr,
		requestC:  make(chan *mcp.MeshConfigRequest),
		responseC: make(chan *WatchResponse),
		watches:   make(map[string]*watch),
		watcher:   s.watcher,
		id:        atomic.AddInt64(&s.nextStreamID, 1),
	}

	var messageNames []string
	for _, typeURL := range s.supportedTypes {
		con.watches[typeURL] = &watch{}

		// extract the message name from the fully qualified type_url.
		if slash := strings.LastIndex(typeURL, "/"); slash >= 0 {
			messageNames = append(messageNames, typeURL[slash+1:])
		}
	}

	scope.Infof("MCP: connection %v: NEW, supported types: %#v", con, messageNames)
	return con, nil
}

// StreamAggregatedResources implements bidirectional streaming method for MCP (ADS).
func (s *Server) StreamAggregatedResources(stream mcp.AggregatedMeshConfigService_StreamAggregatedResourcesServer) error { // nolint: lll
	con, err := s.newConnection(stream)
	if err != nil {
		return err
	}

	defer con.close()
	// 将req加入chan
	go con.receive()

	for {
		select {
		// 由F:\go-source\istio-master\pkg\mcp\snapshot\snapshot.go  170行 异步写入管道的
		case resp, more := <-con.responseC:
			if !more || resp == nil {
				return status.Errorf(codes.Unavailable, "server canceled watch: more=%v resp=%v",
					more, resp)
			}
			if err := con.pushServerResponse(resp); err != nil {
				return err
			}
		case req, more := <-con.requestC:
			if !more {
				return con.reqError
			}
			// 处理MCP  client的请求 从snapshot (distribution)中获取k8s的配置
			if err := con.processClientRequest(req); err != nil {
				return err
			}
		case <-stream.Context().Done():
			scope.Debugf("MCP: connection %v: stream done, err=%v", con, stream.Context().Err())
			return stream.Context().Err()
		}
	}
}

// String implements Stringer.String.
func (con *connection) String() string {
	return fmt.Sprintf("{addr=%v id=%v}", con.peerAddr, con.id)
}

func (con *connection) send(resp *WatchResponse) (string, error) {
	envelopes := make([]mcp.Envelope, 0, len(resp.Envelopes))
	for _, envelope := range resp.Envelopes {
		envelopes = append(envelopes, *envelope)
	}
	msg := &mcp.MeshConfigResponse{
		VersionInfo: resp.Version,
		Envelopes:   envelopes,
		TypeUrl:     resp.TypeURL,
	}

	// increment nonce
	con.streamNonce = con.streamNonce + 1
	msg.Nonce = strconv.FormatInt(con.streamNonce, 10)
	if err := con.stream.Send(msg); err != nil {
		return "", err
	}
	scope.Infof("MCP: connection %v: SEND version=%v nonce=%v", con, resp.Version, msg.Nonce)
	return msg.Nonce, nil
}

func (con *connection) receive() {
	defer close(con.requestC)
	for {
		req, err := con.stream.Recv()
		if err != nil {
			if status.Code(err) == codes.Canceled || err == io.EOF {
				scope.Infof("MCP: connection %v: TERMINATED %q", con, err)
				return
			}
			scope.Errorf("MCP: connection %v: TERMINATED with errors: %v", con, err)

			// Save the stream error prior to closing the stream. The caller
			// should access the error after the channel closure.
			con.reqError = err
			return
		}
		// 加入请求队列
		con.requestC <- req
	}
}

func (con *connection) close() {
	scope.Infof("MCP: connection %v: CLOSED", con)

	for _, watch := range con.watches {
		if watch.cancel != nil {
			watch.cancel()
		}
	}
}


//F:\go-source\istio-master\mixer\pkg\config\mcp\backend.go 作为客户端发请求过来
func (con *connection) processClientRequest(req *mcp.MeshConfigRequest) error {
	watch, ok := con.watches[req.TypeUrl]
	if !ok {
		return status.Errorf(codes.InvalidArgument, "unsupported type_url %q", req.TypeUrl)
	}

	// nonces can be reused across streams; we verify nonce only if nonce is not initialized
	if watch.nonce == "" || watch.nonce == req.ResponseNonce {
		if watch.nonce == "" {
			scope.Debugf("MCP: connection %v: WATCH for %v", con, req.TypeUrl)
		} else {
			scope.Debugf("MCP: connection %v ACK version=%q with nonce=%q",
				con, req.TypeUrl, req.VersionInfo, req.ResponseNonce)
		}

		if watch.cancel != nil {
			watch.cancel()
		}
		var resp *WatchResponse
		// F:\go-source\istio-master\pkg\mcp\snapshot\snapshot.go 获取配置
		resp, watch.cancel = con.watcher.Watch(req, con.responseC)
		if resp != nil {
			// 
			nonce, err := con.send(resp)
			if err != nil {
				return err
			}
			watch.nonce = nonce
		}
	} else {
		scope.Warnf("MCP: connection %v: NACK type_url=%v version=%v with nonce=%q (watch.nonce=%q) error=%#v",
			con, req.TypeUrl, req.VersionInfo, req.ResponseNonce, watch.nonce, req.ErrorDetail)
	}
	return nil
}

func (con *connection) pushServerResponse(resp *WatchResponse) error {
	nonce, err := con.send(resp)
	if err != nil {
		return err
	}

	watch, ok := con.watches[resp.TypeURL]
	if !ok {
		scope.Errorf("MCP: connection %v: internal error: received push response for unsupported type: %v",
			con, resp.TypeURL)
		return status.Errorf(codes.Internal,
			"failed to update internal stream nonce for %v",
			resp.TypeURL)
	}
	watch.nonce = nonce
	return nil
}

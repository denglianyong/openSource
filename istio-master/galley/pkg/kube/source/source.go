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

package source

import (
	"sort"
	"strings"
	"time"

	"github.com/gogo/protobuf/proto"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"

	"istio.io/istio/galley/pkg/kube"
	kube_meta "istio.io/istio/galley/pkg/metadata/kube"

	"istio.io/istio/galley/pkg/runtime"
	"istio.io/istio/galley/pkg/runtime/resource"
	"istio.io/istio/pkg/log"
)

var scope = log.RegisterScope("kube-source", "Source for Kubernetes", 0)

// source is an implementation of runtime.Source.
type sourceImpl struct {
	ifaces kube.Interfaces
	ch     chan resource.Event

	listeners []*listener
}

var _ runtime.Source = &sourceImpl{}

// New returns a Kubernetes implementation of runtime.Source.
func New(k kube.Interfaces, resyncPeriod time.Duration) (runtime.Source, error) {
	return newSource(k, resyncPeriod, kube_meta.Types.All())
}

func newSource(k kube.Interfaces, resyncPeriod time.Duration, specs []kube.ResourceSpec) (runtime.Source, error) {
	s := &sourceImpl{
		ifaces: k,
	}

	sort.Slice(specs, func(i, j int) bool {
		return strings.Compare(specs[i].CanonicalResourceName(), specs[j].CanonicalResourceName()) < 0
	})

	scope.Infof("Registering the following resources:")
	for i, spec := range specs {
		scope.Infof("[%d]", i)
		scope.Infof("  Source:    %s", spec.CanonicalResourceName())
		scope.Infof("  Type URL:  %s", spec.Target.TypeURL)
	//监听k8s的事件，交给process处理
		l, err := newListener(k, resyncPeriod, spec, s.process)
		if err != nil {
			scope.Errorf("Error registering listener: %v", err)
			return nil, err
		}

		s.listeners = append(s.listeners, l)
	}

	return s, nil
}

// Start implements runtime.Source
func (s *sourceImpl) Start() (chan resource.Event, error) {
	s.ch = make(chan resource.Event, 1024)

	for _, l := range s.listeners {
		// 启动监听器从k8s上获取 event  F:\go-source\istio-master\galley\pkg\kube\source\listener.go
		l.start()
	}

	// Wait in a background go-routine until all listeners are synced and send a full-sync event.
	go func() {
		for _, l := range s.listeners {
			l.waitForCacheSync()
		}
		s.ch <- resource.Event{Kind: resource.FullSync}
	}()
// 返回 F:\go-source\istio-master\galley\pkg\runtime\processor.go
	return s.ch, nil
}

// Stop implements runtime.Source
func (s *sourceImpl) Stop() {
	for _, a := range s.listeners {
		a.stop()
	}
}

func (s *sourceImpl) process(l *listener, kind resource.EventKind, key, version string, u *unstructured.Unstructured) {
	var item proto.Message
	var err error
	if u != nil {
		if key, item, err = l.spec.Converter(l.spec.Target, key, u); err != nil {
			scope.Errorf("Unable to convert unstructured to proto: %s/%s", key, version)
			return
		}
	}

	rid := resource.VersionedKey{
		Key: resource.Key{
			TypeURL:  l.spec.Target.TypeURL,
			FullName: key,
		},
		Version: resource.Version(version),
	}

	e := resource.Event{
		ID:   rid,
		Kind: kind,
		Item: item,
	}

	scope.Debugf("Dispatching source event: %v", e)
	//加入管道 
	s.ch <- e
}

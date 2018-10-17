/*
Copyright 2014 The Kubernetes Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

// A binary that can morph into all of the other kubernetes binaries. You can
// also soft-link to it busybox style.
//
// CAUTION: If you update code in this file, you may need to also update code
//          in contrib/mesos/cmd/km/km.go
package main

import (
	"os"

	_ "k8s.io/kubernetes/pkg/client/metrics/prometheus" // for client metric registration
	_ "k8s.io/kubernetes/pkg/version/prometheus"        // for version metric registration
)

func main() {
	hk := HyperKube{
		Name: "hyperkube",
		Long: "This is an all-in-one binary that can run any of the various Kubernetes servers.",
	}
//命令服务器--接受命令??
	hk.AddServer(NewKubectlServer())
	//这里就是api server 接收命令
	hk.AddServer(NewKubeAPIServer())
	// 控制器
	hk.AddServer(NewKubeControllerManager())
	//资源调度
	hk.AddServer(NewScheduler())
	//agent 
	hk.AddServer(NewKubelet())
	//服务代理
	hk.AddServer(NewKubeProxy())
	//Federation servers  集群吗??   这个server不知道干嘛的
	hk.AddServer(NewFederationAPIServer())
	//各种controller  replicatset  cluster  service controller  ===集群相关
	hk.AddServer(NewFederationCMServer())

	hk.RunToExit(os.Args)
}

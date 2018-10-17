// Copyright 2017 Istio Authors
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

package config

import (
	"istio.io/istio/mixer/pkg/config/crd"
	"istio.io/istio/mixer/pkg/config/mcp"
	"istio.io/istio/mixer/pkg/config/store"
)

// StoreInventory returns the inventory of StoreBackend.
// 返回存储后台的清单
//F:\go-source\istio-master\mixer\pkg\server\server.go调用
func StoreInventory() []store.RegisterFunc {
	return []store.RegisterFunc{
		crd.Register,//F:\go-source\istio-master\mixer\pkg\config\crd\init.go
		mcp.Register,// 与mcp server  F:\go-source\istio-master\mixer\pkg\config\mcp\backend.go
	}
}

/*
Copyright 2015 The Kubernetes Authors.

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

package main

import (
	"os"

	"k8s.io/kubernetes/pkg/kubectl/cmd"
	cmdutil "k8s.io/kubernetes/pkg/kubectl/cmd/util"
)

func NewKubectlServer() *Server {
	//创建各种命令
	cmd := cmd.NewKubectlCommand(cmdutil.NewFactory(nil), os.Stdin, os.Stdout, os.Stderr)
	localFlags := cmd.LocalFlags()
	localFlags.SetInterspersed(false)

	return &Server{
		name:        "kubectl",
		SimpleUsage: "Kubernetes command line client",
		Long:        "Kubernetes command line client",
		Run: func(s *Server, args []string) error {
			cmd.SetArgs(args)
			return cmd.Execute()// 执行命令，细节很多，但不需要看
		},
		flags: localFlags,
	}
}

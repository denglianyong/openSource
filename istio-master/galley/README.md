# Galley

Galley is the top-level config ingestion, processing and distribution component of
Istio. It is responsible for insulating the rest of the Istio components from the
details of obtaining user configuration from the underlying platform. It contains 
Kubernetes CRD listeners for collecting configuration, an MCP protocol server
implementation for distributing config, and a validation web-hook for pre-ingestion 
validation by Kubernetes API Server.

Galley配置采集，处理，分发到istio各个组件.作用是将istio组件与底层平台获取用户配置细节隔离.
它包含k8s CRD监听器，用来收集配置，MCP服务器用来实现配置分发，一个web-hook供k8s  api-server来做验证

提供了Istio的配置管理功能，目前还处于研发阶段。

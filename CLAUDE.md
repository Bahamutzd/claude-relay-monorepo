# CLAUDE.md

此文件为 Claude Code (claude.ai/code) 在此代码仓库中工作时提供指导。

## 开发命令

### 核心开发命令
- `npm run dev:frontend` - 启动 Nuxt 前端开发服务器 (localhost:3000)
- `npm run dev:backend` - 使用 Bun 启动 Hono 后端开发服务器 (localhost:8787，更快的启动速度)
- `npm run dev:backend:wrangler` - 使用 Wrangler 启动后端（模拟 Cloudflare Workers 环境）
- `npm run build:all` - 同时构建前端和后端
- `npm run deploy:all` - 构建并部署两个服务到 Cloudflare（先部署后端，再部署前端）

### 单独包命令
- `npm run build:frontend` / `npm run build:backend` - 构建单独的包
- `npm run deploy:frontend` / `npm run deploy:backend` - 部署单独的包
- `npm run type-check` - 对两个包进行 TypeScript 验证

### 代码质量
- `npm run lint` - 对所有 TypeScript/Vue 文件运行 ESLint
- `npm run lint:fix` - 自动修复 ESLint 问题
- `npm run format` - 使用 Prettier 格式化代码
- `npm run format:check` - 检查格式化而不进行修改

## 架构概览

这是一个包含 Cloudflare 全栈应用的 **monorepo**，具有共享的 TypeScript 类型。

### 项目结构
```
├── packages/
│   ├── frontend/          # Nuxt 4 + Tailwind CSS 前端
│   └── backend/           # 用于 Cloudflare Workers 的 Hono 后端
└── shared/                # 共享的 TypeScript 类型和常量
```

### 技术栈
- **前端**: Nuxt 4, Vue 3, Tailwind CSS，部署到 Cloudflare Pages
- **后端**: Hono 框架, TypeScript，部署到 Cloudflare Workers
- **共享**: 前后端共享的 TypeScript 类型和 API 常量

### 关键架构决策

#### 后端 (Hono + Cloudflare Workers)
- **服务分层架构**: 路由层(Routes) → 服务层(Services) → 存储层(KV)
- **智能路由代理**: 自动路由请求到官方 Claude API 或第三方 LLM 供应商
- **格式转换机制**: 内置 Claude-OpenAI 格式双向转换器，支持扩展
- **供应商管理**: 动态配置和管理多个 LLM 供应商
- **统一错误处理**: 全局异常捕获和标准化错误响应
- **开放 CORS 配置**: 支持所有来源访问，适配 Claude Code 客户端
- **TypeScript 类型安全**: 严格类型检查和共享类型定义

#### 前端 (Nuxt 4 + Cloudflare Pages)
- 针对 Cloudflare Pages 配置，具有优化的构建设置
- 使用手动代码分割（vendor chunks）以提升性能
- 首页在构建时预渲染
- 通过 `runtimeConfig` 管理环境变量

#### 共享代码策略
- **类型定义**: 管理相关(`admin.ts`)、通用 API(`api.ts`)、认证(`auth.ts`)
- **常量配置**: 管理中心(`admin.ts`)、API 端点(`endpoints.ts`)、错误码(`errors.ts`)
- **导入路径**: 从 packages 使用相对路径如 `../../../shared/types/admin`

### 环境配置

#### 后端环境变量 (wrangler.toml)
- `NODE_ENV` - 控制错误详细程度 (production/preview)
- `ADMIN_USERNAME` - 管理中心登录用户名 (默认: admin)
- `ADMIN_PASSWORD` - 管理中心登录密码 (默认: password123)
- `CLAUDE_RELAY_ADMIN_KV` - KV Namespace 绑定，存储管理数据和供应商配置

#### 前端环境变量
- `NUXT_PUBLIC_API_BASE_URL` - 后端 API 基础 URL
- `NUXT_PUBLIC_APP_NAME` / `NUXT_PUBLIC_APP_VERSION` - 应用元数据

### 部署流程
1. **后端先部署** (`npm run deploy:backend`) - 确保 API 可用
2. **前端后部署** (`npm run deploy:frontend`) - 连接到线上 API
3. **完整部署** (`npm run deploy:all`) - 处理整个序列

### 开发工作流程
- 前端默认连接到已部署的后端（而非本地后端）
- 两个包都使用基于 workspace 的 npm 脚本以保持一致性
- 共享类型确保前后端 API 契约一致性
- 所有包都启用 TypeScript 严格模式
- 后端作为 Claude Code 请求转发代理，支持来自任意客户端的访问

#### 后端本地开发模式
后端支持两种本地开发模式：
1. **Bun 模式（推荐）** - `npm run dev:backend`
   - 使用 Bun 运行时启动，启动速度快
   - 使用本地 KV 存储（`.kv-storage/` 目录）
   - 支持热重载（`--hot` 标志）
   - 适合快速开发和测试
   
2. **Wrangler 模式** - `npm run dev:backend:wrangler`
   - 使用 Wrangler 模拟完整的 Cloudflare Workers 环境
   - 更接近生产环境的行为
   - 启动较慢但更真实的环境模拟
   - 适合部署前的最终测试

## 后端 API 架构

### 核心 API 端点

#### Claude API 代理 (`/v1/*`)
- `POST /v1/messages` - 代理 Claude API 消息请求，智能路由到官方 Claude 或第三方供应商
- `GET /v1/health` - Claude API 代理服务健康检查

#### 管理中心 API (`/api/admin/*`)
- `POST /api/admin/auth` - 管理员认证登录
- `GET /api/admin/dashboard` - 获取仪表板数据（供应商统计、系统状态等）
- `GET /api/admin/providers` - 获取所有模型供应商
- `POST /api/admin/providers` - 添加新的模型供应商
- `DELETE /api/admin/providers/:id` - 删除模型供应商
- `GET /api/admin/models` - 获取可用模型列表
- `POST /api/admin/select-model` - 选择默认使用的模型

#### 系统端点
- `GET /health` - 应用健康检查和状态信息

### 服务层架构

#### ClaudeProxyService (智能代理服务) 
- **职责**: 智能路由请求到官方 Claude API 或第三方供应商、智能错误处理
- **核心方法**:
  - `proxyRequest(request)` - 根据选中的模型智能路由请求
  - `proxyToOfficialClaude(request)` - 代理到官方 Claude API
  - `proxyToThirdParty(request, provider)` - 代理到第三方供应商
- **特性**: 自动模型选择、智能路由、统一错误处理、流式响应支持

#### LLMProxyService (LLM 代理服务)
- **职责**: 处理第三方 LLM 供应商的请求转发和格式转换
- **核心方法**:
  - `handleRequest(claudeRequest, providerName)` - 处理并转发请求到指定供应商
  - `registerProviderFromConfig(provider)` - 从配置动态注册供应商
  - `getTransformerForProvider(provider)` - 为供应商选择合适的转换器
- **特性**: 动态供应商注册、自动格式转换、支持多种转换器

#### ClaudeToOpenAITransformer (格式转换器)
- **职责**: 在 Claude API 格式和 OpenAI API 格式之间进行双向转换
- **核心方法**:
  - `transformRequestOut(request)` - 将 Claude 请求转换为 OpenAI 格式
  - `transformResponseIn(response)` - 将 OpenAI 响应转换为 Claude 格式
  - `convertStreamToClaudeFormat(stream)` - 转换流式响应格式
- **特性**: 完整的消息格式转换、工具调用支持、流式响应处理

#### AdminService (管理服务)
- **职责**: 管理中心功能实现、模型供应商管理、系统配置
- **核心方法**:
  - `verifyAdmin(username, password)` - 验证管理员凭据
  - `getDashboardData()` - 获取仪表板统计数据
  - `getProviders()` / `addProvider()` / `deleteProvider()` - 供应商管理
  - `getAvailableModels()` / `selectModel()` - 模型选择和切换

### 数据存储设计

#### KV 存储结构
```typescript
// OAuth Token 存储
'claude_token': {
  access_token: string,
  refresh_token: string,
  expires_at: number,
  scope: string,
  // ... 其他 token 信息
}

// 管理中心数据存储
'admin_model_providers': ModelProvider[]  // 模型供应商列表
'admin_selected_model': SelectedModel     // 当前选中的模型
'admin_provider_{{id}}': {               // 各供应商的详细配置
  apiKey: string,
  endpoint: string,
  model: string,
  transformer: string
}
```

### 错误处理机制

#### 自定义错误类型
- `AuthError` - OAuth 认证相关错误
- `TokenExpiredError` - Token 过期错误  
- `ClaudeApiError` - Claude API 调用错误
- `ValidationError` - 请求参数验证错误

#### 统一错误响应格式
```typescript
{
  success: false,
  error: {
    type: 'ERROR_TYPE',
    message: 'Human readable message'
  },
  timestamp: string
}
```

### 智能路由机制

#### 请求路由流程
1. 接收 Claude API 格式的请求
2. 检查当前选中的模型配置
3. 如果是官方 Claude 模型，直接转发到 Claude API
4. 如果是第三方模型：
   - 使用对应的转换器转换请求格式
   - 转发到第三方供应商 API
   - 将响应转换回 Claude 格式
5. 返回统一的 Claude API 格式响应

#### 支持的转换器
- `claude-to-openai` - Claude 格式与 OpenAI 格式双向转换
- 可通过 `LLMProxyService.addTransformer()` 添加自定义转换器

### 需要理解的关键文件

#### 后端核心文件
- `packages/backend/src/index.ts` - Hono 应用入口，中间件和路由配置
- `packages/backend/src/routes/claude.ts` - Claude API 代理路由
- `packages/backend/src/routes/admin.ts` - 管理中心 API 路由
- `packages/backend/src/services/claude.ts` - 智能代理服务实现
- `packages/backend/src/services/llm-proxy.ts` - LLM 代理服务
- `packages/backend/src/services/claude-to-openai-transformer.ts` - 格式转换器
- `packages/backend/src/services/admin.ts` - 管理服务实现
- `packages/backend/src/types/env.ts` - 环境变量和绑定类型定义
- `packages/backend/src/types/transformer.ts` - 转换器类型定义

#### 共享类型和常量
- `shared/types/admin.ts` - 管理中心类型定义
- `shared/types/api.ts` - 通用 API 类型
- `shared/types/auth.ts` - 认证相关类型
- `shared/constants/admin.ts` - 管理中心常量
- `shared/constants/endpoints.ts` - API 端点常量
- `shared/constants/errors.ts` - 错误码定义

#### 配置文件
- `packages/backend/wrangler.toml` - Cloudflare Workers 部署配置
- `根目录/package.json` - Workspace 配置和统一脚本

### 开发最佳实践

#### 代码组织原则
- **分层架构**: 严格的路由→服务→存储分层
- **类型安全**: 全链路 TypeScript 类型约束
- **错误优先**: 完善的错误处理和用户友好提示
- **无状态设计**: 利用 KV 存储实现状态持久化

#### 性能优化策略
- **响应流**: 直接转发流式响应，支持 Claude 和 OpenAI 格式
- **动态路由**: 根据选中模型智能路由，避免不必要的转换
- **并发控制**: 异步处理和适当的超时控制
- **转换优化**: 流式转换，避免全量缓存响应数据

代码库设计重点关注简洁性、类型安全和 Cloudflare 平台深度集成，为 Claude Code 提供智能的多模型代理服务。

## 管理中心功能

### 功能概述

管理中心提供了一个完整的 Web 界面来管理第三方模型供应商配置和模型选择。

#### 核心功能
- **🔐 管理员认证**: 基于环境变量的简单认证机制
- **📊 系统仪表板**: 显示供应商统计、系统状态等信息
- **🔧 模型供应商管理**: 添加、删除第三方 AI 模型供应商（魔搭、智谱 AI、OpenAI 兼容服务等）
- **🎯 模型选择**: 在官方 Claude 和第三方模型间切换默认使用的模型
- **🔄 智能路由**: 自动将请求路由到选中的模型供应商

#### 预设供应商支持
- **魔搭 Qwen**: 阿里云魔搭社区的 Qwen 系列模型
- **智谱 AI**: GLM-4 等先进语言模型  
- **OpenAI Compatible**: 兼容 OpenAI API 的其他服务

### 访问方式

#### 开发环境
- 前端本地开发: `http://localhost:3000/admin`
- 后端本地开发: `http://localhost:8787` (Wrangler 默认端口)
- 本地开发时，前端自动使用 `http://localhost:8787` 作为 API 基础 URL

#### 生产环境
- 管理中心: `https://claude-relay-frontend.pages.dev/admin`
- 默认登录凭据: `admin` / `password123`

### 模型供应商配置

#### 添加供应商流程
1. **选择预设模板**: 选择魔搭、智谱 AI 或 OpenAI 兼容模板
2. **填写配置信息**: 
   - 供应商名称和描述
   - API 端点 URL
   - API 密钥
   - 模型名称
   - 转换器类型（通常为 claude-to-openai）
3. **保存配置**: 供应商配置保存到 KV 存储
4. **选择使用**: 在模型选择页面切换到新添加的供应商

#### 智能路由工作流程
1. Claude Code 发送请求到 `/v1/messages`
2. 系统检查当前选中的模型
3. 如果是第三方模型，使用 LLMProxyService 处理：
   - 转换请求格式
   - 转发到供应商 API
   - 转换响应格式
4. 返回标准 Claude API 格式响应

### 页面结构

```
/                         # 首页（自动重定向到管理中心）
/admin                    # 登录页面
├── /admin/dashboard      # 主仪表板
│   ├── 模型供应商标签页    # 管理第三方AI模型供应商（默认页面）
│   └── 模型选择标签页      # 选择默认使用的模型
└── /admin/add-provider   # 添加供应商页面
```

### 文件组织

#### 前端文件
```
packages/frontend/pages/admin/
├── index.vue           # 登录页面
├── dashboard.vue       # 主仪表板  
└── add-provider.vue    # 添加供应商页面
```

#### 后端文件
```
packages/backend/src/
├── routes/admin.ts     # 管理 API 路由
└── services/admin.ts   # 管理服务实现
```

#### 共享类型
```
shared/
├── types/admin.ts      # 管理中心类型定义
└── constants/admin.ts  # 管理中心常量配置
```

### 部署注意事项

1. **环境变量配置**: 确保在 `wrangler.toml` 中正确设置 `ADMIN_USERNAME` 和 `ADMIN_PASSWORD`
2. **安全考虑**: 生产环境中应修改默认密码
3. **KV 存储**: 管理中心数据存储在 `CLAUDE_RELAY_ADMIN_KV` namespace 中

### 开发建议

- 管理中心采用简化设计，专注核心功能
- 认证机制简单但安全，基于环境变量验证
- UI 设计复用原型，保持与项目整体风格一致
- API 设计遵循现有模式，确保类型安全
- 支持多种 LLM 供应商，提供统一的 Claude API 接口
- 智能路由机制确保请求正确转发到对应的模型供应商
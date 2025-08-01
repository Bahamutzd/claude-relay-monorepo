# CLAUDE.md

此文件为 Claude Code (claude.ai/code) 在此代码仓库中工作时提供指导。

## 开发命令

### 核心开发命令
- `npm run dev:frontend` - 启动 Nuxt 前端开发服务器 (localhost:3000)，使用 `nuxt.config.dev.ts` 开发配置
- `npm run dev:backend` - 使用 Bun 启动 Hono 后端开发服务器 (localhost:8787，更快的启动速度，--hot 热重载)
- `npm run dev:backend:wrangler` - 使用 Wrangler 启动后端（模拟 Cloudflare Workers 环境，端口 8787）
- `npm run build:all` - 同时构建前端和后端
- `npm run deploy:all` - 构建并部署两个服务到 Cloudflare（先部署后端，再部署前端）

### 单独包命令
- `npm run build:frontend` / `npm run build:backend` - 构建单独的包
- `npm run deploy:frontend` / `npm run deploy:backend` - 部署单独的包到 Cloudflare
- `npm run type-check` - 对后端进行 TypeScript 验证（`tsc --noEmit`）

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
│   ├── frontend/          # Nuxt 4 + Vue 3 + Tailwind CSS 前端
│   └── backend/           # Hono + Cloudflare Workers 后端
├── shared/                # 共享的 TypeScript 类型和常量
│   ├── types/            # 类型定义
│   └── constants/        # 常量配置
└── docs/                 # 文档
```

### 技术栈
- **前端**: Nuxt 4, Vue 3, Tailwind CSS, Pinia，部署到 Cloudflare Pages
- **后端**: Hono 框架, TypeScript, Bun 运行时，部署到 Cloudflare Workers
- **共享**: 前后端共享的 TypeScript 类型和 API 常量
- **存储**: Cloudflare KV 存储
- **工具**: ESLint, Prettier, TypeScript

### 关键架构特性

#### 后端 (Hono + Cloudflare Workers)
- **双模式启动**: Bun 模式（快速开发）+ Wrangler 模式（生产环境模拟）
- **服务分层架构**: 路由层(Routes) → 服务层(Services) → 存储层(KV)
- **智能路由代理**: 自动路由请求到官方 Claude API 或第三方 LLM 供应商
- **多格式转换器**: 支持 Claude-OpenAI、Claude-Gemini 格式双向转换
- **供应商管理**: 动态配置和管理多个 LLM 供应商（魔搭、智谱 AI、OpenAI 兼容）
- **统一错误处理**: 全局异常捕获和标准化错误响应
- **开放 CORS 配置**: 支持所有来源访问，适配 Claude Code 客户端
- **定时任务**: 每 6 小时自动刷新 Claude 账号 Token
- **本地存储支持**: 开发模式使用本地 KV 存储，生产模式使用 Cloudflare KV

#### 前端 (Nuxt 4 + Cloudflare Pages)
- **双配置文件**: `nuxt.config.ts`（生产）+ `nuxt.config.dev.ts`（开发）
- **针对 Cloudflare Pages 优化**: preset 配置和构建优化
- **智能代码分割**: 手动 vendor chunks 分割（vue-vendor, pinia）
- **缓存策略**: 路由级缓存控制，静态资源长期缓存
- **开发优化**: 禁用类型检查、压缩等提升开发启动速度
- **环境变量管理**: 开发环境自动连接本地后端，生产环境连接部署的后端

#### 共享代码策略
- **类型定义**: 管理相关(`admin.ts`)、通用 API(`api.ts`)、认证(`auth.ts`)
- **常量配置**: 管理中心(`admin.ts`)、API 端点(`endpoints.ts`)、错误码(`errors.ts`)、OAuth(`oauth.ts`)
- **导入路径**: 从 packages 使用相对路径如 `../../../shared/types/admin`

### 环境配置

#### 后端环境变量 (wrangler.toml)
- `NODE_ENV` - 环境标识 (production/preview/development)
- `ADMIN_USERNAME` / `ADMIN_PASSWORD` - 管理中心登录凭据（通过 Cloudflare Dashboard 设置）
- **KV 绑定**: `CLAUDE_RELAY_ADMIN_KV` - 存储所有管理数据和配置
- **定时任务**: 每 6 小时执行 Token 刷新（`0 */6 * * *`）

#### 前端环境变量
- **生产**: `NUXT_PUBLIC_API_BASE_URL=https://claude-relay-backend.117yelixin.workers.dev`
- **开发**: `NUXT_PUBLIC_API_BASE_URL=http://localhost:8787`
- `NUXT_PUBLIC_APP_NAME` / `NUXT_PUBLIC_APP_VERSION` - 应用元数据

### 部署流程
1. **后端先部署** (`npm run deploy:backend`) - 确保 API 可用
2. **前端后部署** (`npm run deploy:frontend`) - 连接到线上 API
3. **完整部署** (`npm run deploy:all`) - 自动化整个序列

### 开发工作流程
- **前端**: 开发模式连接本地后端 (`localhost:8787`)，生产模式连接部署的后端
- **后端**: 推荐使用 Bun 模式开发，部署前用 Wrangler 模式测试
- **类型安全**: 共享类型确保前后端 API 契约一致性
- **热重载**: 前端和后端都支持热重载开发

#### 后端开发模式详解
1. **Bun 模式（推荐）** - `npm run dev:backend`
   - 入口文件: `src/index.bun.ts`
   - 使用 Bun 运行时，启动速度极快
   - 本地 KV 存储实现（`utils/local-kv-storage.ts`）
   - 支持热重载（`--hot` 标志）
   - 适合快速开发和调试
   
2. **Wrangler 模式** - `npm run dev:backend:wrangler`
   - 入口文件: `src/index.ts`
   - 完整的 Cloudflare Workers 环境模拟
   - 真实的 KV 存储（远程或本地预览）
   - 更接近生产环境行为
   - 适合部署前最终测试

## 后端 API 架构

### 核心 API 端点

#### Claude API 代理 (`/v1/*`)
- `POST /v1/messages` - 智能代理 Claude API，支持官方 Claude 和第三方供应商路由
- `GET /v1/health` - Claude API 代理服务健康检查

#### 管理中心 API (`/api/admin/*`)
- `POST /api/admin/auth` - 管理员认证登录
- `GET /api/admin/dashboard` - 获取仪表板数据（供应商统计、系统状态、账号信息等）
- `GET /api/admin/providers` - 获取所有模型供应商配置
- `POST /api/admin/providers` - 添加新的模型供应商
- `PUT /api/admin/providers/:id` - 编辑模型供应商配置
- `DELETE /api/admin/providers/:id` - 删除模型供应商
- `GET /api/admin/models` - 获取可用模型列表（官方 + 第三方）
- `POST /api/admin/select-model` - 选择默认使用的模型
- `GET /api/admin/claude-accounts` - 获取 Claude 账号列表
- `POST /api/admin/claude-accounts` - 添加新的 Claude 账号
- `DELETE /api/admin/claude-accounts/:id` - 删除 Claude 账号
- `POST /api/admin/claude-accounts/:id/auth` - Claude 账号 OAuth 认证

#### 系统端点
- `GET /health` - 应用健康检查和系统状态信息

### 服务层架构

#### ClaudeProxyService (智能代理服务)
- **文件**: `src/services/claude.ts`
- **职责**: 智能路由请求到官方 Claude API 或第三方供应商
- **核心方法**:
  - `proxyRequest(request)` - 根据选中模型智能路由请求
  - `proxyToOfficialClaude(request)` - 代理到官方 Claude API
  - `proxyToThirdParty(request, provider)` - 代理到第三方供应商
- **特性**: 自动模型选择、智能路由、统一错误处理、流式响应支持

#### LLMProxyService (LLM 代理服务)
- **文件**: `src/services/llm-proxy.ts`
- **职责**: 处理第三方 LLM 供应商的请求转发和格式转换
- **核心方法**:
  - `handleRequest(claudeRequest, provider)` - 处理并转发请求到指定供应商
  - `registerProviderFromConfig(provider)` - 从配置动态注册供应商
  - `getTransformerForProvider(provider)` - 为供应商选择合适的转换器
- **特性**: 动态供应商注册、自动格式转换、支持多种转换器

#### 格式转换器系统
1. **ClaudeToOpenAITransformer** (`src/services/claude-to-openai-transformer.ts`)
   - Claude API ↔ OpenAI API 格式双向转换
   - 支持消息格式、工具调用、流式响应转换
   
2. **ClaudeToGeminiTransformer** (`src/services/claude-to-gemini-transformer.ts`)
   - Claude API ↔ Google Gemini API 格式双向转换
   - 支持 Gemini 特有的消息格式和参数

#### AdminService (管理服务)
- **文件**: `src/services/admin.ts`
- **职责**: 管理中心功能实现、模型供应商管理、Claude 账号管理
- **核心方法**:
  - `verifyAdmin(username, password)` - 验证管理员凭据
  - `getDashboardData()` - 获取仪表板统计数据
  - `getProviders()` / `addProvider()` / `editProvider()` / `deleteProvider()` - 供应商管理
  - `getAvailableModels()` / `selectModel()` - 模型选择和切换
  - `getClaudeAccounts()` / `addClaudeAccount()` / `deleteClaudeAccount()` - Claude 账号管理
  - `refreshClaudeAccountToken()` - 刷新 Claude 账号 Token

### 数据存储设计

#### KV 存储结构
```typescript
// Claude 账号管理
'claude_accounts': ClaudeAccount[]           // Claude 账号列表
'claude_account_{{id}}_token': OAuthToken   // 各账号的 OAuth Token

// 模型供应商管理
'admin_model_providers': ModelProvider[]    // 模型供应商列表
'admin_selected_model': SelectedModel       // 当前选中的模型

// 供应商配置（敏感信息）
'admin_provider_{{id}}': {
  apiKey: string,
  endpoint: string,
  model: string,
  transformer: string
}
```

### 错误处理机制

#### 自定义错误类型 (`src/utils/errors.ts`)
- `AuthError` - OAuth 认证相关错误
- `TokenExpiredError` - Token 过期错误  
- `ClaudeApiError` - Claude API 调用错误
- `ValidationError` - 请求参数验证错误
- `ProviderError` - 第三方供应商错误

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
1. 接收 Claude API 格式的请求到 `/v1/messages`
2. 检查当前选中的模型配置
3. **官方 Claude 模型**: 直接转发到 Claude API
4. **第三方模型**:
   - 根据供应商类型选择转换器
   - 转换请求格式（Claude → OpenAI/Gemini）
   - 转发到第三方供应商 API
   - 转换响应格式（OpenAI/Gemini → Claude）
5. 返回统一的 Claude API 格式响应

#### 支持的转换器
- `claude-to-openai` - 兼容 OpenAI API 的供应商（魔搭、智谱 AI 等）
- `claude-to-gemini` - Google Gemini API 供应商
- 可扩展：通过 `LLMProxyService.addTransformer()` 添加自定义转换器

### 关键文件结构

#### 后端核心文件
```
packages/backend/src/
├── index.ts                                   # Workers 入口文件
├── index.bun.ts                              # Bun 开发模式入口
├── routes/
│   ├── claude.ts                             # Claude API 代理路由
│   └── admin.ts                              # 管理中心 API 路由
├── services/
│   ├── claude.ts                             # 智能代理服务
│   ├── llm-proxy.ts                          # LLM 代理服务
│   ├── claude-to-openai-transformer.ts      # OpenAI 格式转换器
│   ├── claude-to-gemini-transformer.ts      # Gemini 格式转换器
│   └── admin.ts                              # 管理服务
├── middleware/
│   ├── result-handler.ts                     # 统一错误处理中间件
│   └── kv-validator.ts                       # KV 绑定验证中间件
├── utils/
│   ├── errors.ts                             # 自定义错误类型
│   ├── response.ts                           # 响应工具函数
│   └── local-kv-storage.ts                  # 本地 KV 存储实现
└── types/
    ├── env.ts                                # 环境变量和绑定类型
    └── transformer.ts                        # 转换器类型定义
```

#### 前端文件结构
```
packages/frontend/
├── nuxt.config.ts                            # 生产配置
├── nuxt.config.dev.ts                       # 开发配置
├── pages/
│   ├── index.vue                             # 首页（重定向到管理中心）
│   └── admin/
│       ├── index.vue                         # 管理中心登录页
│       ├── dashboard.vue                     # 主仪表板
│       └── add-provider.vue                  # 添加供应商页面
├── components/
│   └── ProviderForm.vue                      # 供应商表单组件
└── assets/css/
    └── main.css                              # Tailwind CSS 样式
```

#### 共享代码
```
shared/
├── types/
│   ├── admin.ts                              # 管理中心类型定义
│   ├── api.ts                                # 通用 API 类型
│   └── auth.ts                               # 认证相关类型
└── constants/
    ├── admin.ts                              # 管理中心常量
    ├── endpoints.ts                          # API 端点常量
    ├── errors.ts                             # 错误码定义
    └── oauth.ts                              # OAuth 相关常量
```

### 开发最佳实践

#### 代码组织原则
- **分层架构**: 严格的路由→服务→存储分层，职责明确
- **类型安全**: 全链路 TypeScript 类型约束，共享类型定义
- **错误优先**: 完善的错误处理和用户友好的错误信息
- **无状态设计**: 利用 KV 存储实现状态持久化，支持多实例部署

#### 性能优化策略
- **流式响应**: 直接转发流式响应，支持 Claude 和 OpenAI 格式
- **智能路由**: 根据选中模型动态路由，避免不必要的转换
- **并发控制**: 异步处理和适当的超时控制
- **缓存优化**: 前端静态资源缓存，API 响应缓存策略

## 管理中心功能

### 功能概述

管理中心提供了一个完整的 Web 界面来管理 Claude 账号、第三方模型供应商配置和模型选择。

#### 核心功能
- **🔐 管理员认证**: 基于环境变量的安全认证机制
- **👤 Claude 账号管理**: 添加、删除 Claude 账号，OAuth 认证，Token 自动刷新
- **📊 系统仪表板**: 显示账号状态、供应商统计、系统健康状态等信息
- **🔧 模型供应商管理**: 添加、编辑、删除第三方 AI 模型供应商
- **🎯 模型选择**: 在官方 Claude 和第三方模型间切换默认使用的模型
- **🔄 智能路由**: 自动将请求路由到选中的模型供应商

#### 预设供应商支持
- **魔搭 Qwen**: 阿里云魔搭社区的 Qwen 系列模型（claude-to-openai 转换器）
- **智谱 AI**: GLM-4 等先进语言模型（claude-to-openai 转换器）
- **Google Gemini**: Google 的 Gemini 系列模型（claude-to-gemini 转换器）
- **OpenAI Compatible**: 兼容 OpenAI API 的其他服务

### 访问方式

#### 开发环境
- **前端开发**: `http://localhost:3000/admin`
- **后端 API**: `http://localhost:8787`
- **开发配置**: 前端自动连接本地后端 API

#### 生产环境
- **管理中心**: `https://claude-relay-frontend.pages.dev/admin`
- **后端 API**: `https://claude-relay-backend.117yelixin.workers.dev`
- **默认凭据**: 通过 Cloudflare Dashboard 环境变量设置

### Claude 账号管理

#### Claude 账号功能
1. **添加账号**: 创建新的 Claude 账号配置
2. **OAuth 认证**: 通过 Claude 官方 OAuth 流程获取访问令牌
3. **Token 管理**: 自动刷新过期的访问令牌
4. **状态监控**: 实时显示账号状态（活跃/不活跃/过期）

#### OAuth 认证流程
1. 在管理中心添加新的 Claude 账号
2. 系统生成 OAuth 认证链接（PKCE 安全流程）
3. 用户访问链接完成 Claude 官方授权
4. 系统接收授权码并交换访问令牌
5. Token 存储到 KV 并设置自动刷新

### 模型供应商配置

#### 添加供应商流程
1. **选择供应商类型**: OpenAI 兼容或 Google Gemini
2. **选择预设模板**: 魔搭、智谱 AI、Google Gemini 或自定义
3. **填写配置信息**: 
   - 供应商名称和描述
   - API 端点 URL
   - API 密钥
   - 模型名称
   - 转换器类型（自动选择或手动指定）
4. **保存配置**: 供应商配置安全存储到 KV
5. **选择使用**: 在模型选择页面切换到新添加的供应商

#### 智能路由工作流程
1. 客户端发送请求到 `/v1/messages`
2. 系统检查当前选中的模型配置
3. **官方 Claude**: 直接转发到 Claude API
4. **第三方供应商**: 
   - 使用对应转换器转换请求格式
   - 转发到供应商 API
   - 转换响应格式返回
5. 返回标准 Claude API 格式响应

### 页面结构

```
/                         # 首页（自动重定向到管理中心）
/admin                    # 管理中心登录页面
├── /admin/dashboard      # 主仪表板
│   ├── Claude 账号标签页  # Claude 账号管理（默认页面）
│   ├── 模型供应商标签页    # 第三方 AI 模型供应商管理
│   └── 模型选择标签页      # 选择默认使用的模型
└── /admin/add-provider   # 添加供应商页面
```

### 部署配置

#### 环境变量设置（Cloudflare Dashboard）
```bash
# 管理员凭据（必须在 Dashboard 中设置，不要写在 wrangler.toml）
ADMIN_USERNAME=your_admin_username
ADMIN_PASSWORD=your_secure_password

# 可选：Claude OAuth 应用配置
CLAUDE_CLIENT_ID=your_client_id
CLAUDE_CLIENT_SECRET=your_client_secret
```

#### KV 命名空间
- **绑定名**: `CLAUDE_RELAY_ADMIN_KV`
- **用途**: 存储 Claude 账号、供应商配置、系统设置等所有数据
- **配置**: 在 `wrangler.toml` 中已正确配置

### 系统特性

#### 安全特性
- 环境变量存储敏感信息，不在代码中硬编码
- OAuth PKCE 流程确保认证安全
- API 密钥加密存储在 KV 中
- CORS 配置允许跨域访问

#### 自动化特性
- 定时任务自动刷新 Claude 账号 Token
- 智能错误处理和用户友好的错误信息
- 实时状态监控和健康检查
- 响应式设计，支持移动端访问

#### 扩展性特性
- 插件化转换器系统，支持添加新的 API 格式
- 模块化架构，易于添加新的供应商类型
- 类型安全的配置管理，减少配置错误
- 统一的 API 接口，支持多种客户端

代码库设计重点关注安全性、可维护性和用户体验，为 Claude Code 和其他客户端提供稳定可靠的多模型代理服务。
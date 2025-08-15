/**
 * Claude to OpenAI 转换器
 * 将 Claude API 格式转换为 OpenAI 兼容格式，支持官方和第三方 OpenAI 兼容 API
 * 使用官方 OpenAI SDK 实现，提供类型安全、错误处理和流式支持
 */

import type { Transformer } from './base-transformer'
import type { 
  MessageCreateParamsBase,
  Message,
  MessageParam,
  TextBlockParam,
  ImageBlockParam,
  ToolUseBlockParam,
  ToolResultBlockParam,
  Tool as ClaudeTool,
  StopReason
} from '@anthropic-ai/sdk/resources/messages'
import OpenAI from 'openai'
import type { 
  ChatCompletionCreateParamsNonStreaming,
  ChatCompletionCreateParamsStreaming,
  ChatCompletionMessageParam,
  ChatCompletionTool,
  ChatCompletionToolChoiceOption,
  ChatCompletion
} from 'openai/resources/chat/completions'
import { fixToolCallArguments, fixSingleQuoteJson, fixStreamingToolArgument } from '../../../utils/json-tools'

export class ClaudeToOpenAITransformer implements Transformer {
  private client: OpenAI | null = null
  private baseURL: string = ''

  /**
   * 初始化 OpenAI 客户端
   * 支持官方 OpenAI API 和第三方兼容 API（如 Azure OpenAI、Ollama、vLLM、LocalAI 等）
   */
  public initializeClient(apiKey: string, options?: { 
    baseUrl?: string
  }): void {
    if (!options?.baseUrl) {
      throw new Error('baseUrl is required for OpenAI-compatible providers')
    }
    
    this.baseURL = options.baseUrl
    this.client = new OpenAI({
      apiKey,
      baseURL: this.baseURL
      // 使用 OpenAI SDK 默认配置：10分钟超时，2次重试
    })
  }

  /**
   * 获取客户端实例
   */
  private getClient(): OpenAI {
    if (!this.client) {
      throw new Error('OpenAI client not initialized. Call initializeClient() first.')
    }
    return this.client
  }

  /**
   * 主要转换方法 - 直接调用 OpenAI SDK 并转换响应
   */
  async processRequest(claudeRequest: MessageCreateParamsBase, model: string): Promise<Message | ReadableStream> {
    const client = this.getClient()
    
    // 记录原始 Claude 请求
    // logClaudeRequest(claudeRequest)

    if (claudeRequest.stream) {
      // 流式响应
      const streamParams = this.buildStreamingParams(claudeRequest, model)
      
      const stream = await client.chat.completions.create(streamParams)
      
      return await this.transformStreamResponse(stream)
    } else {
      // 非流式响应
      const params = this.buildNonStreamingParams(claudeRequest, model)
      
      const response = await client.chat.completions.create(params)      
      
      const claudeResponse = this.transformResponse(response)
      
      return claudeResponse
    }
  }

  /**
   * 构建非流式请求参数
   */
  private buildNonStreamingParams(claudeRequest: MessageCreateParamsBase, model: string): ChatCompletionCreateParamsNonStreaming {
    const baseParams = this.buildBaseParams(claudeRequest, model)
    return {
      ...baseParams,
      stream: false
    }
  }

  /**
   * 构建流式请求参数
   */
  private buildStreamingParams(claudeRequest: MessageCreateParamsBase, model: string): ChatCompletionCreateParamsStreaming {
    const baseParams = this.buildBaseParams(claudeRequest, model)
    return {
      ...baseParams,
      stream: true
    }
  }

  /**
   * 构建基础请求参数
   */
  private buildBaseParams(claudeRequest: MessageCreateParamsBase, model: string) {
    const params: Omit<ChatCompletionCreateParamsNonStreaming, 'stream'> = {
      model,
      messages: this.transformMessages(claudeRequest.messages || [], claudeRequest.system)
    }

    // 基础参数转换
    if (claudeRequest.max_tokens) params.max_completion_tokens = claudeRequest.max_tokens
    if (claudeRequest.temperature !== undefined) params.temperature = claudeRequest.temperature
    if (claudeRequest.top_p !== undefined) params.top_p = claudeRequest.top_p
    if (claudeRequest.stop_sequences) {
      params.stop = claudeRequest.stop_sequences.length === 1 
        ? claudeRequest.stop_sequences[0] 
        : claudeRequest.stop_sequences
    }

    // 工具转换
    if (claudeRequest.tools?.length) {
      params.tools = this.transformTools(claudeRequest.tools as ClaudeTool[])
      if (claudeRequest.tool_choice) {
        params.tool_choice = this.transformToolChoice(claudeRequest.tool_choice)
      }
    }

    return params
  }

  /**
   * 转换消息数组
   */
  private transformMessages(messages: MessageParam[], system?: string | Array<any>): ChatCompletionMessageParam[] {
    const openaiMessages: ChatCompletionMessageParam[] = []

    // 添加系统消息
    if (system) {
      const systemContent = typeof system === 'string' 
        ? system 
        : this.extractTextFromContent(system)
      
      openaiMessages.push({
        role: 'system',
        content: systemContent
      })
    }

    // 转换用户和助手消息
    for (const message of messages) {
      const openaiMessage = this.transformMessage(message)
      if (openaiMessage) {
        openaiMessages.push(openaiMessage)
      }
    }

    return openaiMessages
  }

  /**
   * 转换单个消息
   */
  private transformMessage(message: MessageParam): ChatCompletionMessageParam | null {
    const role = message.role === 'assistant' ? 'assistant' : 'user'
    
    if (typeof message.content === 'string') {
      return {
        role: role as 'user' | 'assistant',
        content: message.content
      }
    }

    if (Array.isArray(message.content)) {
      const content: Array<any> = []
      const toolCalls: Array<any> = []

      for (const item of message.content) {
        switch (item.type) {
          case 'text':
            const textBlock = item as TextBlockParam
            content.push({
              type: 'text',
              text: textBlock.text
            })
            break

          case 'image':
            const imageBlock = item as ImageBlockParam
            if (imageBlock.source.type === 'base64') {
              content.push({
                type: 'image_url',
                image_url: {
                  url: `data:${imageBlock.source.media_type};base64,${imageBlock.source.data}`,
                  detail: 'auto' as const
                }
              })
            }
            break

          case 'tool_use':
            const toolUseBlock = item as ToolUseBlockParam
            
            // 验证工具名称是否存在
            if (!toolUseBlock.name || typeof toolUseBlock.name !== 'string') {
              console.warn('Tool use block missing or invalid name:', toolUseBlock)
              break
            }
            
            toolCalls.push({
              id: toolUseBlock.id, // 直接使用 Claude ID
              type: 'function' as const,
              function: {
                name: toolUseBlock.name,
                arguments: JSON.stringify(toolUseBlock.input || {})
              }
            })
            break

          case 'tool_result':
            const toolResultBlock = item as ToolResultBlockParam
            
            return {
              role: 'tool',
              content: typeof toolResultBlock.content === 'string' 
                ? toolResultBlock.content 
                : JSON.stringify(toolResultBlock.content),
              tool_call_id: toolResultBlock.tool_use_id // 直接使用 Claude ID
            }
        }
      }

      const openaiMessage: ChatCompletionMessageParam = {
        role: role as 'user' | 'assistant',
        content: content.length > 0 ? content : ''
      }

      if (toolCalls.length > 0 && role === 'assistant') {
        (openaiMessage as any).tool_calls = toolCalls
      }

      return openaiMessage
    }

    return null
  }

  /**
   * 转换工具定义
   */
  private transformTools(tools: ClaudeTool[]): ChatCompletionTool[] {
    return tools.map(tool => ({
      type: 'function' as const,
      function: {
        name: tool.name,
        description: tool.description,
        parameters: this.cleanupParameters(tool.input_schema)
      }
    }))
  }

  /**
   * 转换工具选择策略
   */
  private transformToolChoice(toolChoice: MessageCreateParamsBase['tool_choice']): ChatCompletionToolChoiceOption {
    if (typeof toolChoice === 'string') {
      return toolChoice === 'none' ? 'none' : 'auto'
    }
    
    if (toolChoice && typeof toolChoice === 'object') {
      if (toolChoice.type === 'tool' && 'name' in toolChoice) {
        return {
          type: 'function',
          function: { name: toolChoice.name }
        }
      }
      return toolChoice.type === 'auto' ? 'auto' : 'none'
    }
    
    return 'auto'
  }

  /**
   * 转换 OpenAI 响应为 Claude 格式
   */
  private transformResponse(response: ChatCompletion): Message {
    const choice = response.choices[0]
    const content: any[] = []

    // 处理文本内容
    if (choice.message.content) {
      content.push({
        type: 'text',
        text: choice.message.content
      })
    }

    // 处理工具调用
    if (choice.message.tool_calls) {
      for (const toolCall of choice.message.tool_calls) {
        content.push({
          type: 'tool_use',
          id: toolCall.id, // 直接使用 OpenAI ID
          name: toolCall.function.name,
          input: fixToolCallArguments(toolCall.function.arguments || '{}')
        })
      }
    }

    return {
      id: `msg_${Date.now()}`,
      type: 'message',
      role: 'assistant',
      model: response.model,
      content,
      stop_reason: this.mapFinishReason(choice.finish_reason),
      stop_sequence: null,
      usage: {
        input_tokens: response.usage?.prompt_tokens || 0,
        output_tokens: response.usage?.completion_tokens || 0,
        cache_creation_input_tokens: null,
        cache_read_input_tokens: null,
        server_tool_use: null,
        service_tier: null
      }
    }
  }

  /**
   * 转换流式响应为 Claude 格式
   * 混合模式：检测到工具调用时切换为非流式处理，避免流式/非流式混合导致的问题
   */
  private async transformStreamResponse(openaiStream: AsyncIterable<OpenAI.Chat.Completions.ChatCompletionChunk>): Promise<ReadableStream> {
    const encoder = new TextEncoder()
    const self = this
    let hasToolCalls = false
    let accumulatedChunks: OpenAI.Chat.Completions.ChatCompletionChunk[] = []
    
    return new ReadableStream({
      async start(controller) {
        try {
          // 第一阶段：收集chunks并检测工具调用
          for await (const chunk of openaiStream) {
            accumulatedChunks.push(chunk)
            
            // 检测是否有工具调用
            if (chunk.choices[0]?.delta?.tool_calls && !hasToolCalls) {
              hasToolCalls = true
              console.log('检测到工具调用，切换为非流式处理模式')
            }
          }

          // 第二阶段：根据是否有工具调用选择处理策略
          if (hasToolCalls) {
            // 有工具调用：使用非流式处理逻辑
            console.log('使用非流式模式处理工具调用')
            
            // 重构完整的ChatCompletion响应
            const completeResponse = self.reconstructCompletionFromChunks(accumulatedChunks)
            
            // 使用现有的非流式转换逻辑（包含工具调用修复）
            const claudeMessage = self.transformResponse(completeResponse)
            
            // 将Claude响应转换为流式事件输出
            self.outputClaudeMessageAsStream(claudeMessage, controller, encoder)
            
          } else {
            // 无工具调用：正常流式处理
            console.log('无工具调用，使用正常流式模式')
            self.outputChunksAsStream(accumulatedChunks, controller, encoder)
          }

        } catch (error) {
          console.error('流式处理发生错误:', error)
          self.handleStreamError(error, controller, encoder)
        } finally {
          controller.close()
        }
      }
    })
  }

  /**
   * 从chunks重构完整的ChatCompletion响应
   */
  private reconstructCompletionFromChunks(chunks: OpenAI.Chat.Completions.ChatCompletionChunk[]): ChatCompletion {
    if (chunks.length === 0) {
      throw new Error('没有chunks可以重构')
    }

    const firstChunk = chunks[0]
    let content = ''
    const toolCalls: any[] = []
    const toolCallsMap: Map<number, any> = new Map()
    let finishReason: string | null = null

    // 合并所有chunks的内容
    for (const chunk of chunks) {
      const choice = chunk.choices[0]
      if (!choice) continue

      // 合并文本内容
      if (choice.delta.content) {
        content += choice.delta.content
      }

      // 合并工具调用
      if (choice.delta.tool_calls) {
        for (const toolCall of choice.delta.tool_calls) {
          const index = toolCall.index || 0
          
          if (!toolCallsMap.has(index)) {
            toolCallsMap.set(index, {
              id: toolCall.id || `tool_${index}`,
              type: 'function' as const,
              function: {
                name: '',
                arguments: ''
              }
            })
          }

          const existingToolCall = toolCallsMap.get(index)!
          
          if (toolCall.function?.name) {
            existingToolCall.function.name = toolCall.function.name
          }
          
          if (toolCall.function?.arguments) {
            existingToolCall.function.arguments += toolCall.function.arguments
          }
        }
      }

      // 记录完成原因
      if (choice.finish_reason) {
        finishReason = choice.finish_reason
      }
    }

    // 转换工具调用映射为数组
    for (const [index, toolCall] of toolCallsMap) {
      toolCalls[index] = toolCall
    }

    // 构建完整响应
    const completeResponse: ChatCompletion = {
      id: `chatcmpl_${Date.now()}`,
      object: 'chat.completion',
      created: Math.floor(Date.now() / 1000),
      model: firstChunk.model,
      choices: [{
        index: 0,
        message: {
          role: 'assistant',
          content: content || null,
          tool_calls: toolCalls.length > 0 ? toolCalls : undefined,
          refusal: null
        },
        finish_reason: finishReason as any,
        logprobs: null
      }],
      usage: {
        prompt_tokens: 0,
        completion_tokens: 0,
        total_tokens: 0
      }
    }

    return completeResponse
  }

  /**
   * 将Claude响应作为流式事件输出
   */
  private outputClaudeMessageAsStream(
    claudeMessage: Message, 
    controller: ReadableStreamDefaultController,
    encoder: TextEncoder
  ): void {
    try {
      // 发送 message_start 事件
      controller.enqueue(encoder.encode(this.createSSEEvent('message_start', {
        type: 'message_start',
        message: claudeMessage
      })))

      let contentIndex = 0

      // 处理所有内容块
      for (const contentBlock of claudeMessage.content) {
        if (contentBlock.type === 'text') {
          // 文本块
          controller.enqueue(encoder.encode(this.createSSEEvent('content_block_start', {
            type: 'content_block_start',
            index: contentIndex,
            content_block: contentBlock
          })))

          // 发送完整文本
          controller.enqueue(encoder.encode(this.createSSEEvent('content_block_delta', {
            type: 'content_block_delta',
            index: contentIndex,
            delta: { type: 'text_delta', text: contentBlock.text }
          })))

          controller.enqueue(encoder.encode(this.createSSEEvent('content_block_stop', {
            type: 'content_block_stop',
            index: contentIndex
          })))

        } else if (contentBlock.type === 'tool_use') {
          // 工具使用块
          controller.enqueue(encoder.encode(this.createSSEEvent('content_block_start', {
            type: 'content_block_start',
            index: contentIndex,
            content_block: contentBlock
          })))

          // 发送完整的工具参数（已修复）
          const inputJson = JSON.stringify(contentBlock.input || {})
          controller.enqueue(encoder.encode(this.createSSEEvent('content_block_delta', {
            type: 'content_block_delta',
            index: contentIndex,
            delta: {
              type: 'input_json_delta',
              partial_json: inputJson
            }
          })))

          controller.enqueue(encoder.encode(this.createSSEEvent('content_block_stop', {
            type: 'content_block_stop',
            index: contentIndex
          })))
        }

        contentIndex++
      }

      // 发送消息完成
      controller.enqueue(encoder.encode(this.createSSEEvent('message_delta', {
        type: 'message_delta',
        delta: {
          stop_reason: claudeMessage.stop_reason,
          stop_sequence: claudeMessage.stop_sequence
        }
      })))

      // 发送结束事件
      controller.enqueue(encoder.encode(this.createSSEEvent('message_stop', {
        type: 'message_stop'
      })))

    } catch (error) {
      console.error('输出Claude消息流时出错:', error)
      throw error
    }
  }

  /**
   * 将chunks作为正常流式输出（无工具调用情况）
   */
  private outputChunksAsStream(
    chunks: OpenAI.Chat.Completions.ChatCompletionChunk[],
    controller: ReadableStreamDefaultController,
    encoder: TextEncoder
  ): void {
    let messageStarted = false
    let contentStarted = false
    let contentIndex = 0

    for (const chunk of chunks) {
      // 发送 message_start 事件
      if (!messageStarted) {
        controller.enqueue(encoder.encode(this.createSSEEvent('message_start', {
          type: 'message_start',
          message: {
            id: `msg_${Date.now()}`,
            type: 'message',
            role: 'assistant',
            model: chunk.model,
            content: [],
            stop_reason: null,
            stop_sequence: null,
            usage: { input_tokens: 0, output_tokens: 0 }
          }
        })))
        messageStarted = true
      }

      const choice = chunk.choices[0]
      if (!choice) continue

      // 处理文本内容
      if (choice.delta.content) {
        // 如果是第一次收到内容，发送 content_block_start
        if (!contentStarted) {
          controller.enqueue(encoder.encode(this.createSSEEvent('content_block_start', {
            type: 'content_block_start',
            index: contentIndex,
            content_block: { type: 'text', text: '' }
          })))
          contentStarted = true
        }

        // 发送内容增量
        controller.enqueue(encoder.encode(this.createSSEEvent('content_block_delta', {
          type: 'content_block_delta',
          index: contentIndex,
          delta: { type: 'text_delta', text: choice.delta.content }
        })))
      }

      // 处理完成
      if (choice.finish_reason) {
        if (contentStarted) {
          controller.enqueue(encoder.encode(this.createSSEEvent('content_block_stop', {
            type: 'content_block_stop',
            index: contentIndex
          })))
        }

        // 发送消息完成
        controller.enqueue(encoder.encode(this.createSSEEvent('message_delta', {
          type: 'message_delta',
          delta: {
            stop_reason: this.mapFinishReason(choice.finish_reason),
            stop_sequence: null
          }
        })))

        // 发送结束事件
        controller.enqueue(encoder.encode(this.createSSEEvent('message_stop', {
          type: 'message_stop'
        })))
      }
    }
  }

  /**
   * 处理流式错误
   */
  private handleStreamError(
    error: any,
    controller: ReadableStreamDefaultController,
    encoder: TextEncoder
  ): void {
    try {
      // 发送基本的错误恢复事件
      controller.enqueue(encoder.encode(this.createSSEEvent('message_start', {
        type: 'message_start',
        message: {
          id: `msg_${Date.now()}`,
          type: 'message',
          role: 'assistant',
          model: 'unknown',
          content: [],
          stop_reason: 'error',
          stop_sequence: null,
          usage: { input_tokens: 0, output_tokens: 0 }
        }
      })))
      
      controller.enqueue(encoder.encode(this.createSSEEvent('message_delta', {
        type: 'message_delta',
        delta: {
          stop_reason: 'error',
          stop_sequence: null
        }
      })))
      
      controller.enqueue(encoder.encode(this.createSSEEvent('message_stop', {
        type: 'message_stop'
      })))
    } catch (recoveryError) {
      console.error('错误恢复也失败:', recoveryError)
    }
    
    // 通知控制器错误
    const errorMessage = error instanceof Error ? error.message : String(error)
    controller.error(new Error(`流式处理失败: ${errorMessage}`))
  }

  /**
   * 映射完成原因
   */
  private mapFinishReason(reason: string | null): StopReason {
    if (!reason) return 'end_turn'
    
    const mapping: Record<string, StopReason> = {
      'stop': 'end_turn',
      'length': 'max_tokens',
      'tool_calls': 'tool_use',
      'content_filter': 'end_turn'
    }
    
    return mapping[reason] || 'end_turn'
  }


  /**
   * 清理参数定义
   */
  private cleanupParameters(params: any): any {
    if (!params || typeof params !== 'object') return params
    
    const cleaned = JSON.parse(JSON.stringify(params))
    this.removeUnsupportedProperties(cleaned)
    return cleaned
  }

  /**
   * 递归移除不支持的属性
   */
  private removeUnsupportedProperties(obj: any): void {
    if (!obj || typeof obj !== 'object') return
    
    if (Array.isArray(obj)) {
      obj.forEach(item => this.removeUnsupportedProperties(item))
      return
    }

    // 移除 OpenAI 不支持但 Claude 可能包含的属性
    delete obj.$schema
    delete obj.const

    // 递归处理子属性
    Object.values(obj).forEach(value => this.removeUnsupportedProperties(value))
  }

  /**
   * 从复合内容中提取文本
   */
  private extractTextFromContent(content: Array<TextBlockParam | ImageBlockParam>): string {
    return content
      .filter((item): item is TextBlockParam => item.type === 'text')
      .map(item => item.text)
      .join('\n')
  }

  /**
   * 清理资源
   */
  public cleanup(): void {
    // 无需清理，因为不再维护映射
  }

  /**
   * 创建 SSE 事件格式
   */
  private createSSEEvent(event: string, data: Record<string, any>): string {
    return `event: ${event}\ndata: ${JSON.stringify(data)}\n\n`
  }
}
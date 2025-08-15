import { describe, it, expect } from 'vitest'
import { ClaudeToOpenAITransformer } from '../../../src/services/proxy/transformers/claude-to-openai'

describe('ClaudeToOpenAI缓存工具调用测试', () => {
  describe('流式工具调用缓存机制', () => {
    it('应该缓存工具调用内容并在完成后一次性输出', async () => {
      const transformer = new ClaudeToOpenAITransformer()
      transformer.initializeClient('test-key', { baseUrl: 'https://api.openai.com/v1' })
      
      // 模拟包含工具调用的Claude请求
      const claudeRequest = {
        model: 'claude-3-sonnet-20240229',
        messages: [
          {
            role: 'user' as const,
            content: '创建一个测试任务'
          }
        ],
        tools: [
          {
            name: 'create_task',
            description: '创建一个新任务',
            input_schema: {
              type: 'object',
              properties: {
                title: { type: 'string' },
                description: { type: 'string' }
              }
            }
          }
        ],
        stream: true,
        max_tokens: 1000
      }
      
      // 模拟OpenAI流式响应，包含工具调用和单引号JSON
      async function* mockOpenAIStream() {
        // 开始消息
        yield {
          id: 'chatcmpl-test',
          object: 'chat.completion.chunk' as const,
          created: Date.now(),
          model: 'gpt-4',
          choices: [{
            index: 0,
            delta: { role: 'assistant' as const },
            finish_reason: null
          }]
        }
        
        // 文本内容
        yield {
          id: 'chatcmpl-test',
          object: 'chat.completion.chunk' as const,
          created: Date.now(),
          model: 'gpt-4',
          choices: [{
            index: 0,
            delta: { content: '我来为你创建一个测试任务。' },
            finish_reason: null
          }]
        }
        
        // 工具调用开始（包含单引号JSON）
        yield {
          id: 'chatcmpl-test',
          object: 'chat.completion.chunk' as const,
          created: Date.now(),
          model: 'gpt-4',
          choices: [{
            index: 0,
            delta: {
              tool_calls: [{
                index: 0,
                id: 'call_test',
                type: 'function' as const,
                function: {
                  name: 'create_task',
                  arguments: "{'title': '"
                }
              }]
            },
            finish_reason: null
          }]
        }
        
        // 工具调用参数继续（单引号问题）
        yield {
          id: 'chatcmpl-test',
          object: 'chat.completion.chunk' as const,
          created: Date.now(),
          model: 'gpt-4',
          choices: [{
            index: 0,
            delta: {
              tool_calls: [{
                index: 0,
                function: {
                  arguments: "测试任务', 'description': '这是一个测试任务'}"
                }
              }]
            },
            finish_reason: null
          }]
        }
        
        // 完成
        yield {
          id: 'chatcmpl-test',
          object: 'chat.completion.chunk' as const,
          created: Date.now(),
          model: 'gpt-4',
          choices: [{
            index: 0,
            delta: {},
            finish_reason: 'tool_calls' as const
          }]
        }
      }
      
      // 模拟transformer的processRequest方法直接调用transformStreamResponse
      // 创建模拟的OpenAI客户端响应
      const mockStream = transformer as any
      mockStream.client = {
        chat: {
          completions: {
            create: () => mockOpenAIStream()
          }
        }
      }
      
      // 获取转换后的流
      const stream = await transformer.processRequest(claudeRequest, 'gpt-4')
      
      if (!(stream instanceof ReadableStream)) {
        throw new Error('Expected ReadableStream')
      }
      
      // 读取所有流式事件
      const reader = stream.getReader()
      const events: string[] = []
      
      try {
        while (true) {
          const { done, value } = await reader.read()
          if (done) break
          
          const text = new TextDecoder().decode(value)
          events.push(text)
        }
      } finally {
        reader.releaseLock()
      }
      
      const allEvents = events.join('')
      
      // 验证事件顺序和内容
      // 1. 应该包含message_start
      expect(allEvents).toContain('event: message_start')
      
      // 2. 应该包含文本内容的流式输出
      expect(allEvents).toContain('我来为你创建一个测试任务')
      
      // 3. 应该包含工具调用，且参数已被修复
      expect(allEvents).toContain('event: content_block_start')
      expect(allEvents).toContain('"type":"tool_use"')
      expect(allEvents).toContain('"name":"create_task"')
      
      // 4. 最重要：应该包含修复后的JSON（双引号）
      expect(allEvents).toContain('"title":"测试任务"')
      expect(allEvents).toContain('"description":"这是一个测试任务"')
      
      // 5. 不应该包含原始的单引号格式
      expect(allEvents).not.toContain("'title':")
      expect(allEvents).not.toContain("'description':")
      
      // 6. 应该包含完成事件
      expect(allEvents).toContain('event: message_stop')
    })
  })
})
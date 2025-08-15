import { describe, it, expect } from 'vitest'
import { fixSingleQuoteJson, safeParseJson, fixToolCallArguments, fixStreamingToolArgument } from '../../../src/utils/json-tools'

describe('JSON修复工具', () => {
  describe('fixSingleQuoteJson', () => {
    it('应该修复简单的单引号对象', () => {
      const input = "{'key': 'value'}"
      const expected = '{"key": "value"}'
      expect(fixSingleQuoteJson(input)).toBe(expected)
    })

    it('应该修复复杂的单引号对象', () => {
      const input = "{'name': 'test', 'age': 25, 'active': true}"
      const result = fixSingleQuoteJson(input)
      const parsed = JSON.parse(result)
      expect(parsed).toEqual({ name: 'test', age: 25, active: true })
    })

    it('应该处理嵌套对象', () => {
      const input = "{'user': {'name': 'john', 'details': {'age': 30}}}"
      const result = fixSingleQuoteJson(input)
      const parsed = JSON.parse(result)
      expect(parsed).toEqual({
        user: {
          name: 'john',
          details: { age: 30 }
        }
      })
    })

    it('应该处理数组', () => {
      const input = "{'items': ['one', 'two', 'three']}"
      const result = fixSingleQuoteJson(input)
      const parsed = JSON.parse(result)
      expect(parsed).toEqual({ items: ['one', 'two', 'three'] })
    })

    it('应该保持已经正确的JSON不变', () => {
      const input = '{"key": "value"}'
      expect(fixSingleQuoteJson(input)).toBe(input)
    })

    it('应该处理空对象', () => {
      const input = "{}"
      expect(fixSingleQuoteJson(input)).toBe(input)
    })

    it('应该处理非JSON字符串', () => {
      const input = "not a json"
      expect(fixSingleQuoteJson(input)).toBe(input)
    })
  })

  describe('safeParseJson', () => {
    it('应该解析有效的JSON', () => {
      const input = '{"key": "value"}'
      const result = safeParseJson(input)
      expect(result).toEqual({ key: 'value' })
    })

    it('应该修复并解析单引号JSON', () => {
      const input = "{'key': 'value'}"
      const result = safeParseJson(input)
      expect(result).toEqual({ key: 'value' })
    })

    it('应该在解析失败时返回默认值', () => {
      const input = "invalid json"
      const defaultValue = { error: true }
      const result = safeParseJson(input, defaultValue)
      expect(result).toEqual(defaultValue)
    })

    it('应该在没有默认值时返回空对象', () => {
      const input = "invalid json"
      const result = safeParseJson(input)
      expect(result).toEqual({})
    })
  })

  describe('fixToolCallArguments', () => {
    it('应该处理工具调用参数', () => {
      const input = "{'command': 'ls -la', 'directory': '/home'}"
      const result = fixToolCallArguments(input)
      expect(result).toEqual({ 
        command: 'ls -la',
        directory: '/home'
      })
    })

    it('应该处理复杂的工具调用参数', () => {
      const input = "{'function': 'analyze_code', 'params': {'file': 'main.py', 'checks': ['syntax', 'style']}}"
      const result = fixToolCallArguments(input)
      expect(result).toEqual({
        function: 'analyze_code',
        params: {
          file: 'main.py',
          checks: ['syntax', 'style']
        }
      })
    })

    it('应该在无效输入时返回空对象', () => {
      const input = "invalid"
      const result = fixToolCallArguments(input)
      expect(result).toEqual({})
    })

    it('应该处理空字符串', () => {
      const input = ""
      const result = fixToolCallArguments(input)
      expect(result).toEqual({})
    })
  })

  describe('fixStreamingToolArgument', () => {
    it('应该修复流式片段中的单引号', () => {
      const fragment = "{'key': 'value'}"
      const result = fixStreamingToolArgument(fragment)
      expect(result).toBe('{"key": "value"}')
    })

    it('应该处理累积参数的修复', () => {
      const accumulated = "{'name': 'test'"
      const fragment = ", 'age': 25}"
      const result = fixStreamingToolArgument(fragment, accumulated)
      // 应该能识别并修复完整的参数
      expect(result).toContain('"')
    })

    it('应该保持正确格式的片段不变', () => {
      const fragment = '{"key": "value"}'
      const result = fixStreamingToolArgument(fragment)
      expect(result).toBe(fragment) // 没有单引号，应该保持不变
    })

    it('应该处理部分片段', () => {
      const fragment = "'command': 'ls -la'"
      const result = fixStreamingToolArgument(fragment)
      expect(result).toBe('"command": "ls -la"')
    })

    it('应该处理空片段', () => {
      const fragment = ''
      const result = fixStreamingToolArgument(fragment)
      expect(result).toBe('')
    })

    it('应该处理非JSON字符串片段', () => {
      const fragment = 'not json'
      const result = fixStreamingToolArgument(fragment)
      expect(result).toBe('not json')
    })

    it('应该处理流式工具调用场景', () => {
      // 模拟流式接收工具调用参数的过程
      const fragments = [
        "{'function':",
        " 'analyze_code',",
        " 'params': {",
        "'file': 'main.py'",
        "}}"
      ]
      
      let accumulated = ''
      const results = []
      
      for (const fragment of fragments) {
        const fixed = fixStreamingToolArgument(fragment, accumulated)
        results.push(fixed)
        accumulated += fragment
      }
      
      // 至少应该修复单引号问题
      const finalResult = results.join('')
      expect(finalResult).toContain('"function"')
      expect(finalResult).toContain('"analyze_code"')
    })
  })

  describe('真实场景测试', () => {
    it('应该处理智谱AI返回的单引号格式', () => {
      // 模拟智谱AI可能返回的格式
      const input = "{'query': 'SELECT * FROM users WHERE age > 25', 'limit': 10}"
      const result = fixToolCallArguments(input)
      expect(result).toEqual({
        query: 'SELECT * FROM users WHERE age > 25',
        limit: 10
      })
    })

    it('应该处理Qwen返回的单引号格式', () => {
      // 模拟Qwen可能返回的格式
      const input = "{'action': 'search', 'keywords': ['javascript', 'react'], 'options': {'sort': 'relevance'}}"
      const result = fixToolCallArguments(input)
      expect(result).toEqual({
        action: 'search',
        keywords: ['javascript', 'react'],
        options: { sort: 'relevance' }
      })
    })

    it('应该处理包含特殊字符的参数', () => {
      const input = "{'code': 'function test() { return \\'hello\\'; }', 'language': 'javascript'}"
      const result = fixToolCallArguments(input)
      expect(result.language).toBe('javascript')
      expect(typeof result.code).toBe('string')
    })
  })
})
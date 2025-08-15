/**
 * JSON 工具函数
 * 处理第三方供应商返回的非标准JSON格式（如单引号）
 */

/**
 * 修复单引号JSON字符串为标准双引号格式
 * 处理常见的JSON格式问题，特别是工具调用参数中的单引号问题
 * 
 * @param jsonStr 可能包含单引号的JSON字符串
 * @returns 标准双引号JSON字符串
 */
export function fixSingleQuoteJson(jsonStr: string): string {
  if (!jsonStr || typeof jsonStr !== 'string') {
    return jsonStr
  }

  // 如果已经是有效的JSON，直接返回
  try {
    JSON.parse(jsonStr)
    return jsonStr
  } catch {
    // 继续进行修复
  }

  // 移除首尾空白字符
  let fixed = jsonStr.trim()

  // 如果不是以 { 或 [ 开头，可能不是JSON，直接返回
  if (!fixed.startsWith('{') && !fixed.startsWith('[')) {
    return jsonStr
  }

  try {
    // 使用更强大的正则表达式修复常见的单引号问题
    
    // 先处理已经转义的单引号，避免重复处理
    fixed = fixed.replace(/\\'/g, '\u0001') // 临时标记已转义的单引号
    
    // 修复各种单引号情况
    // 1. 属性名: 'key' -> "key"
    fixed = fixed.replace(/(\s*)'([^']*)'(\s*):/g, '$1"$2"$3:')
    
    // 2. 字符串值: :'value' -> :"value"
    fixed = fixed.replace(/:(\s*)'([^']*)'/g, ':$1"$2"')
    
    // 3. 数组中的字符串: ['value'] -> ["value"]
    fixed = fixed.replace(/\[(\s*)'([^']*)'/g, '[$1"$2"')
    fixed = fixed.replace(/,(\s*)'([^']*)'/g, ',$1"$2"')
    
    // 4. 处理更复杂的嵌套情况
    fixed = fixed.replace(/\{(\s*)'([^']*)'/g, '{$1"$2"')
    
    // 恢复已转义的单引号为转义的双引号
    fixed = fixed.replace(/\u0001/g, '\\"')
    
    // 验证修复后的JSON
    JSON.parse(fixed)
    return fixed
  } catch (error) {
    // 如果修复失败，尝试更激进的修复方法
    try {
      // 最后手段：尝试全局替换（但要小心字符串内容）
      if (fixed.startsWith('{') && fixed.endsWith('}')) {
        let aggressiveFix = fixed
        // 将所有单引号替换为双引号
        aggressiveFix = aggressiveFix.replace(/'/g, '"')
        
        // 尝试解析
        JSON.parse(aggressiveFix)
        return aggressiveFix
      }
    } catch {
      // 最激进的方法也失败了
    }
    
    // 如果所有修复都失败，返回原始字符串
    console.warn('JSON修复失败，返回原始字符串:', jsonStr)
    return jsonStr
  }
}

/**
 * 安全解析JSON字符串，自动修复单引号问题
 * 
 * @param jsonStr JSON字符串
 * @param defaultValue 解析失败时的默认值
 * @returns 解析后的对象或默认值
 */
export function safeParseJson<T = any>(jsonStr: string, defaultValue: T = {} as T): T {
  if (!jsonStr || typeof jsonStr !== 'string') {
    return defaultValue
  }

  try {
    return JSON.parse(jsonStr)
  } catch {
    try {
      const fixed = fixSingleQuoteJson(jsonStr)
      // 只有在实际修复了内容时才尝试解析
      if (fixed !== jsonStr) {
        return JSON.parse(fixed)
      }
    } catch {
      // 修复解析失败，使用默认值
    }
    return defaultValue
  }
}

/**
 * 修复工具调用参数字符串
 * 专门用于处理LLM返回的工具调用参数格式问题
 * 
 * @param argsStr 工具调用参数字符串
 * @returns 修复后的参数对象
 */
export function fixToolCallArguments(argsStr: string): Record<string, any> {
  return safeParseJson(argsStr, {})
}

/**
 * 修复流式工具调用参数片段
 * 处理流式输出中的单引号JSON片段，提供强大的回退机制
 * 
 * @param fragment 参数片段字符串
 * @param accumulated 已累积的参数字符串
 * @returns 修复后的片段字符串
 */
export function fixStreamingToolArgument(fragment: string, accumulated: string = ''): string {
  if (!fragment || typeof fragment !== 'string') {
    return fragment
  }

  try {
    // 首先检查是否需要修复（只有包含单引号的才修复）
    if (fragment.includes("'")) {
      let fixed = fragment.replace(/'/g, '"')
      // 处理已转义的双引号（原本是转义的单引号）
      fixed = fixed.replace(/\\"/g, "\\'")
      return fixed
    }

    // 如果没有单引号，检查是否是JSON片段
    if (accumulated) {
      const testStr = accumulated + fragment
      try {
        // 如果累积的字符串可以解析为JSON，说明片段是正确的
        JSON.parse(testStr)
        return fragment
      } catch {
        // 如果不能解析，尝试修复累积的字符串
        const fixedAccumulated = fixSingleQuoteJson(testStr)
        if (fixedAccumulated !== testStr) {
          // 如果修复成功，计算修复后的片段
          const newFragment = fixedAccumulated.slice(accumulated.length)
          return newFragment || fragment
        }
      }
    }

    return fragment
  } catch (error) {
    console.warn('流式工具参数修复失败:', { fragment, accumulated, error })
    return fragment
  }
}
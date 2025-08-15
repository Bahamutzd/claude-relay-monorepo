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
    // 使用正则表达式修复常见的单引号问题
    // 1. 将属性名的单引号替换为双引号
    // 2. 将字符串值的单引号替换为双引号
    // 3. 处理转义字符
    
    // 先处理已经转义的单引号，避免重复处理
    fixed = fixed.replace(/\\'/g, '\u0001') // 临时标记已转义的单引号
    
    // 匹配属性名: 'key' -> "key"
    fixed = fixed.replace(/(\s*)'([^']*)'(\s*):/g, '$1"$2"$3:')
    
    // 匹配字符串值: :'value' -> :"value" (考虑嵌套引号的情况)
    fixed = fixed.replace(/:(\s*)'([^']*)'/g, ':$1"$2"')
    
    // 匹配数组中的字符串: ['value'] -> ["value"]
    fixed = fixed.replace(/\[(\s*)'([^']*)'/g, '[$1"$2"')
    fixed = fixed.replace(/,(\s*)'([^']*)'/g, ',$1"$2"')
    
    // 恢复已转义的单引号为转义的双引号
    fixed = fixed.replace(/\u0001/g, '\\"')
    
    // 验证修复后的JSON
    JSON.parse(fixed)
    return fixed
  } catch (error) {
    // 如果修复失败，尝试更激进的修复方法
    try {
      // 最后手段：尝试eval方法（仅用于对象字面量）
      if (fixed.startsWith('{') && fixed.endsWith('}')) {
        // 将单引号属性名和值都替换为双引号
        let evalFix = fixed
        // 替换所有单引号为双引号，但要小心字符串内容
        evalFix = evalFix.replace(/'/g, '"')
        JSON.parse(evalFix)
        return evalFix
      }
    } catch {
      // 最后都失败了，返回原始字符串
    }
    
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
      return JSON.parse(fixed)
    } catch {
      return defaultValue
    }
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
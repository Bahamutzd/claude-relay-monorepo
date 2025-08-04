import { ref } from 'vue'
import type { ModelProvider, EditProviderRequest } from '../../../shared/types/admin/providers'
import type { KeyPoolStats } from '../../../shared/types/key-pool'
import { API_ENDPOINTS } from '../../../shared/constants/endpoints'
import { useKeyPool } from './useKeyPool'

export const useProviders = () => {
  const config = useRuntimeConfig()
  const { getMultipleKeyPoolStats, formatKeyPoolStatus } = useKeyPool()
  
  // 响应式数据
  const providers = ref<ModelProvider[]>([])
  const keyPoolStatuses = ref<Map<string, string>>(new Map())
  const expandedProviders = ref<Set<string>>(new Set())
  const showEditModal = ref(false)
  const editingProvider = ref<ModelProvider | null>(null)
  const editLoading = ref(false)
  
  // 确认对话框状态
  const showConfirmDialog = ref(false)
  const confirmDialogConfig = ref<{
    title: string
    message: string
    description?: string
    type: 'danger' | 'warning' | 'info'
    onConfirm: () => void
  } | null>(null)
  const confirmLoading = ref(false)

  // 通知函数
  const showNotification = (message: string, type: 'success' | 'error' | 'info' = 'info') => {
    const notification = document.createElement('div')
    notification.className = `fixed top-4 right-4 z-50 px-6 py-3 rounded-xl text-white font-medium shadow-lg transform translate-x-full transition-transform duration-300 ${
      type === 'success' ? 'bg-emerald-500' : type === 'error' ? 'bg-red-500' : 'bg-orange-500'
    }`
    notification.textContent = message
    document.body.appendChild(notification)
    
    setTimeout(() => {
      notification.classList.remove('translate-x-full')
    }, 100)
    
    setTimeout(() => {
      notification.classList.add('translate-x-full')
      setTimeout(() => {
        document.body.removeChild(notification)
      }, 300)
    }, 3000)
  }

  // 供应商管理方法
  const loadProviders = async () => {
    try {
      const response = await $fetch<{ success: boolean; data: ModelProvider[] }>(
        API_ENDPOINTS.ADMIN_PROVIDERS,
        { baseURL: config.public.apiBaseUrl }
      )
      if (response.success) {
        providers.value = response.data
        // 加载所有供应商的密钥池状态
        await loadKeyPoolStatuses()
      }
    } catch (error) {
      console.error('Failed to load providers:', error)
    }
  }

  const loadKeyPoolStatuses = async () => {
    if (providers.value.length === 0) return
    
    const providerIds = providers.value.map(p => p.id)
    const statsMap = await getMultipleKeyPoolStats(providerIds)
    
    // 更新状态显示
    keyPoolStatuses.value.clear()
    providers.value.forEach(provider => {
      const stats = statsMap.get(provider.id) || null
      keyPoolStatuses.value.set(provider.id, formatKeyPoolStatus(stats))
    })
  }

  const editProvider = (provider: ModelProvider) => {
    editingProvider.value = { ...provider }
    showEditModal.value = true
  }

  const updateProvider = async (request: EditProviderRequest) => {
    if (!editingProvider.value) return
    
    editLoading.value = true
    
    try {
      const response = await $fetch<{ success: boolean; data: ModelProvider }>(
        `${API_ENDPOINTS.ADMIN_PROVIDERS}/${editingProvider.value.id}`,
        {
          method: 'PUT',
          baseURL: config.public.apiBaseUrl,
          body: request
        }
      )
      
      if (response.success) {
        await loadProviders() // 刷新供应商列表
        showEditModal.value = false
        editingProvider.value = null
        showNotification('供应商更新成功', 'success')
      }
    } catch (error) {
      console.error('Failed to update provider:', error)
      showNotification('更新供应商失败', 'error')
    } finally {
      editLoading.value = false
    }
  }

  const cancelEdit = () => {
    showEditModal.value = false
    editingProvider.value = null
  }

  const deleteProvider = async (id: string) => {
    const provider = providers.value.find(p => p.id === id)
    if (!provider) return
    
    confirmDialogConfig.value = {
      title: '删除供应商',
      message: `确定要删除供应商"${provider.name}"吗？`,
      description: '删除后将无法恢复，该供应商的所有配置和密钥池数据也将被清除。',
      type: 'danger',
      onConfirm: async () => {
        confirmLoading.value = true
        
        try {
          const response = await $fetch<{ success: boolean }>(
            `${API_ENDPOINTS.ADMIN_PROVIDERS}/${id}`,
            {
              method: 'DELETE',
              baseURL: config.public.apiBaseUrl
            }
          )
          
          if (response.success) {
            await loadProviders()
            showNotification('删除供应商成功', 'success')
            showConfirmDialog.value = false
          }
        } catch (error) {
          console.error('Failed to delete provider:', error)
          showNotification('删除供应商失败', 'error')
        } finally {
          confirmLoading.value = false
        }
      }
    }
    
    showConfirmDialog.value = true
  }
  
  // 确认对话框方法
  const handleConfirmDialogCancel = () => {
    showConfirmDialog.value = false
    confirmDialogConfig.value = null
    confirmLoading.value = false
  }
  
  const handleConfirmDialogConfirm = () => {
    if (confirmDialogConfig.value?.onConfirm) {
      confirmDialogConfig.value.onConfirm()
    }
  }

  // 展开/收起供应商模型列表
  const toggleProviderExpansion = (providerId: string) => {
    if (expandedProviders.value.has(providerId)) {
      expandedProviders.value.delete(providerId)
    } else {
      expandedProviders.value.add(providerId)
    }
  }

  // 初始化时加载数据
  loadProviders()

  return {
    // 响应式数据
    providers,
    keyPoolStatuses,
    expandedProviders,
    showEditModal,
    editingProvider,
    editLoading,
    
    // 确认对话框状态
    showConfirmDialog,
    confirmDialogConfig,
    confirmLoading,
    
    // 方法
    loadProviders,
    loadKeyPoolStatuses,
    editProvider,
    updateProvider,
    cancelEdit,
    deleteProvider,
    toggleProviderExpansion,
    handleConfirmDialogCancel,
    handleConfirmDialogConfirm
  }
}
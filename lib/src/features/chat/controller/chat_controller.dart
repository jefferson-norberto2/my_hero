import 'dart:typed_data';
import 'package:my_hero/src/core/constants/app_constants.dart';
import 'package:my_hero/src/core/utils/url_utils.dart';
import 'package:my_hero/src/features/chat/data/adapter/chat_attachment.dart';
import 'package:my_hero/src/features/chat/data/adapter/chat_message_adapter.dart';
import 'package:my_hero/src/features/chat/data/adapter/chat_metrics.dart';
import 'package:my_hero/src/features/chat/data/adapter/conversation_model.dart';
import 'package:my_hero/src/features/chat/data/repository/chat_repository.dart';
import 'package:my_hero/src/features/settings/data/models/settings_args.dart';
import 'package:my_hero/src/shared/models/ollama_model.dart';
import 'package:my_hero/src/shared/models/response_style.dart';
import 'package:my_hero/src/shared/services/media_picker_service.dart';
import 'package:signals_flutter/signals_flutter.dart';

/// Manages reactive state and business logic for the chat screen.
class ChatController {
  final ChatRepository _repository;
  final MediaPickerService _mediaPickerService;

  ChatController({
    required ChatRepository repository,
    required MediaPickerService mediaPickerService,
  })  : _repository = repository,
        _mediaPickerService = mediaPickerService;

  // Reactive state signals
  final ollamaBaseUrl = signal<String>(AppConstants.ollamaBaseUrl);
  final availableModels = signal<List<OllamaModel>>([]);
  final selectedModelName = signal<String>('');
  final isOllamaConnected = signal<bool>(true);
  final isModelLoaded = signal<bool>(false);
  final isLoading = signal<bool>(false);
  final isGenerating = signal<bool>(false);
  final loadingProgress = signal<double>(0.0);
  final loadingStatus = signal<String>('');
  final responseStyle = signal<ResponseStyle>(ResponseStyle.simple);
  final chatHistory = signal<List<ChatMessage>>([]);
  final errorMessage = signal<String?>(null);
  final attachedAttachment = signal<ChatAttachment?>(null);
  final attachedImage = signal<Uint8List?>(null);

  // Context and VRAM metrics
  final currentMetrics = signal<ChatMetrics>(const ChatMetrics());

  // Conversation history
  final conversations = signal<List<Conversation>>([]);
  final currentConversationId = signal<String?>(null);

  bool _stopGenerationFlag = false;

  Conversation? get currentConversation {
    final id = currentConversationId.value;
    if (id == null) return null;
    try {
      return conversations.value.firstWhere((c) => c.id == id);
    } catch (_) {
      return null;
    }
  }

  /// Builds route arguments reflecting the active chat state for the settings screen.
  SettingsArgs buildSettingsArgs() {
    return SettingsArgs(
      selectedModelName: selectedModelName.value,
      responseStyle: responseStyle.value,
      ollamaBaseUrl: ollamaBaseUrl.value,
      availableModels: List<OllamaModel>.from(availableModels.value),
      isOllamaConnected: isOllamaConnected.value,
      isModelLoaded: isModelLoaded.value,
    );
  }

  Future<void> init() async {
    await loadSavedConversations();
    await loadOllamaServerUrl();
    await refreshModels();
  }

  Future<void> loadOllamaServerUrl() async {
    try {
      final savedUrl = await _repository.loadOllamaBaseUrl();
      ollamaBaseUrl.value = savedUrl;
    } catch (_) {
      // Keep default URL if persisted configuration cannot be read.
    }
  }

  Future<bool> testOllamaConnection(String host) async {
    final normalized = normalizeOllamaUrl(host);
    return _repository.checkOllamaHealth(normalized);
  }

  Future<bool> updateOllamaHost(String newHost) async {
    final normalized = normalizeOllamaUrl(newHost);
    ollamaBaseUrl.value = normalized;
    _repository.updateOllamaBaseUrl(normalized);
    await _repository.saveOllamaBaseUrl(normalized);
    await refreshModels();
    return isOllamaConnected.value;
  }

  Future<void> refreshModels() async {
    try {
      final isHealthy = await _repository.checkOllamaHealth();
      isOllamaConnected.value = isHealthy;

      if (!isHealthy) {
        errorMessage.value =
            'Ollama não detectado em ${ollamaBaseUrl.value}. Verifique se o serviço está ativo e acessível na rede.';
        return;
      }

      final models = await _repository.getAvailableModels();
      availableModels.value = models;

      if (models.isNotEmpty) {
        // Prefer any model already loaded in Ollama memory.
        final alreadyLoaded = models.where((m) => m.isLoaded).firstOrNull;

        if (selectedModelName.value.isEmpty) {
          final target = alreadyLoaded?.name ??
              (currentConversation?.modelName.isNotEmpty == true
                  ? currentConversation!.modelName
                  : models.first.name);
          selectedModelName.value = target;
        }

        // Check whether the currently selected model is already loaded.
        final currentModelInfo =
            models.where((m) => m.name == selectedModelName.value).firstOrNull;
        if (currentModelInfo != null && currentModelInfo.isLoaded) {
          isModelLoaded.value = true;
          _updateMetricsFromModel(currentModelInfo);
        }
      }
    } catch (e) {
      isOllamaConnected.value = false;
      errorMessage.value = 'Erro ao consultar modelos do Ollama: $e';
    }
  }

  void _updateMetricsFromModel(OllamaModel model) {
    currentMetrics.value = currentMetrics.value.copyWith(
      vramBytes: model.sizeVram,
      contextLimit: model.contextLength > 0 ? model.contextLength : 4096,
    );
  }

  Future<void> loadSavedConversations() async {
    try {
      final saved = await _repository.getConversations();
      if (saved.isNotEmpty) {
        conversations.value = saved;
        await selectConversation(saved.first.id);
      } else {
        await createNewConversation();
      }
    } catch (_) {
      await createNewConversation();
    }
  }

  /// Creates a new conversation with an empty message history.
  Future<void> createNewConversation() async {
    final now = DateTime.now();
    final modelToUse = selectedModelName.value.isNotEmpty
        ? selectedModelName.value
        : (availableModels.value.isNotEmpty
            ? availableModels.value.first.name
            : '');

    final newConv = Conversation(
      id: now.millisecondsSinceEpoch.toString(),
      title: 'Nova Conversa',
      messages: [],
      modelName: modelToUse,
      responseStyle: responseStyle.value,
      createdAt: now,
      updatedAt: now,
    );

    conversations.value = [newConv, ...conversations.value];
    currentConversationId.value = newConv.id;
    chatHistory.value = [];

    currentMetrics.value = currentMetrics.value.copyWith(
      promptTokens: 0,
      evalTokens: 0,
    );

    await _saveCurrentState();
  }

  /// Selects a conversation and restores its messages, style, and token metrics.
  Future<void> selectConversation(String id) async {
    if (isGenerating.value) return;

    final conv = conversations.value.where((c) => c.id == id).firstOrNull;
    if (conv == null) return;

    currentConversationId.value = conv.id;
    chatHistory.value = List.from(conv.messages);
    if (conv.modelName.isNotEmpty) {
      selectedModelName.value = conv.modelName;
    }
    responseStyle.value = conv.responseStyle;

    currentMetrics.value = currentMetrics.value.copyWith(
      promptTokens: conv.lastPromptTokens,
      evalTokens: conv.lastEvalTokens,
    );
  }

  Future<void> deleteConversation(String id) async {
    if (isGenerating.value) return;

    final updated = conversations.value.where((c) => c.id != id).toList();
    conversations.value = updated;

    if (currentConversationId.value == id) {
      if (updated.isNotEmpty) {
        await selectConversation(updated.first.id);
      } else {
        await createNewConversation();
      }
    }

    await _repository.saveConversations(conversations.value);
  }

  /// Preloads the selected model into Ollama memory.
  Future<void> loadModel(String modelName) async {
    isLoading.value = true;
    loadingProgress.value = 0.0;
    loadingStatus.value = 'Solicitando carregamento de $modelName no Ollama...';
    errorMessage.value = null;

    try {
      await _repository.loadModel(
        modelName: modelName,
        onProgress: (progress, status) {
          loadingProgress.value = progress;
          loadingStatus.value = status;
        },
      );

      selectedModelName.value = modelName;
      isModelLoaded.value = true;
      isLoading.value = false;

      // Refresh available models to reflect memory allocation and context limit.
      await refreshModels();

      final modelInfo =
          availableModels.value.where((m) => m.name == modelName).firstOrNull;
      if (modelInfo != null) {
        _updateMetricsFromModel(modelInfo);
      }

      final conv = currentConversation;
      if (conv != null) {
        conv.modelName = modelName;
        await _saveCurrentState();
      }

      _appendSystemMessage(
        'Sistema: Modelo "$modelName" carregado na memória do computador!',
      );
      await _saveCurrentState();
    } catch (e) {
      isLoading.value = false;
      errorMessage.value = 'Falha ao carregar o modelo no Ollama: $e';
    }
  }

  /// Applies saved settings returned by the dedicated settings screen.
  Future<void> applySettings(SettingsResult result) async {
    if (isLoading.value || isGenerating.value) return;

    if (result.ollamaBaseUrl != ollamaBaseUrl.value) {
      ollamaBaseUrl.value = result.ollamaBaseUrl;
      _repository.updateOllamaBaseUrl(result.ollamaBaseUrl);
    }

    isOllamaConnected.value = result.isOllamaConnected;
    availableModels.value = result.availableModels;

    final previousModel = selectedModelName.value;
    final previousStyle = responseStyle.value;
    final wasLoaded = isModelLoaded.value;

    final targetModel = result.selectedModelName;
    final targetStyle = result.responseStyle;

    final modelChanged = targetModel != previousModel;
    final styleChanged = targetStyle != previousStyle;

    selectedModelName.value = targetModel;
    responseStyle.value = targetStyle;

    final conv = currentConversation;
    if (conv != null) {
      conv.modelName = targetModel;
      conv.responseStyle = targetStyle;
    }

    final modelInfo =
        availableModels.value.where((m) => m.name == targetModel).firstOrNull;
    final isAlreadyLoadedInMemory =
        result.isModelLoaded || (modelInfo?.isLoaded ?? false);

    if (!result.isOllamaConnected || targetModel.isEmpty) {
      isModelLoaded.value = false;
    } else if (modelChanged || (!wasLoaded && isAlreadyLoadedInMemory)) {
      if (isAlreadyLoadedInMemory) {
        isModelLoaded.value = true;
        if (modelInfo != null) {
          _updateMetricsFromModel(modelInfo);
        }
        _appendSystemMessage(
          'Sistema: Modelo "$targetModel" carregado na memória do computador!',
        );
      } else if (modelChanged) {
        await loadModel(targetModel);
      }
    } else if (modelInfo != null && modelInfo.isLoaded) {
      isModelLoaded.value = true;
      _updateMetricsFromModel(modelInfo);
    }

    if (styleChanged) {
      _appendSystemMessage(
        'Sistema: Estilo de resposta alterado para "${targetStyle.label}".',
      );
    }

    await _saveCurrentState();
  }

  /// Updates the active conversation model and response style.
  Future<void> updateConversationSettings({
    required String modelName,
    required ResponseStyle newResponseStyle,
  }) async {
    if (isLoading.value || isGenerating.value) return;

    final modelChanged = modelName != selectedModelName.value;
    final styleChanged = newResponseStyle != responseStyle.value;

    if (!modelChanged && !styleChanged) return;

    responseStyle.value = newResponseStyle;
    selectedModelName.value = modelName;

    final conv = currentConversation;
    if (conv != null) {
      conv.modelName = modelName;
      conv.responseStyle = newResponseStyle;
      await _saveCurrentState();
    }

    if (modelChanged) {
      await loadModel(modelName);
    } else if (styleChanged) {
      _appendSystemMessage(
        'Sistema: Estilo de resposta alterado para "${newResponseStyle.label}".',
      );
      await _saveCurrentState();
    }
  }

  void _appendSystemMessage(String text) {
    chatHistory.value = [
      ...chatHistory.value,
      ChatMessage(text: text, isUser: false),
    ];
    _syncMessagesToCurrentConversation();
  }

  void setAttachment(ChatAttachment? attachment) {
    attachedAttachment.value = attachment;
    attachedImage.value =
        attachment?.type == AttachmentType.image ? attachment?.bytes : null;
  }

  void clearAttachment() {
    attachedAttachment.value = null;
    attachedImage.value = null;
  }

  void setAttachedImage(Uint8List? bytes) {
    if (bytes == null) {
      clearAttachment();
    } else {
      setAttachment(
        ChatAttachment(
          name: 'imagem.png',
          type: AttachmentType.image,
          bytes: bytes,
        ),
      );
    }
  }

  Future<void> pickImage({bool fromCamera = false}) async {
    try {
      final attachment = fromCamera
          ? await _mediaPickerService.pickImageFromCamera()
          : await _mediaPickerService.pickImageFromGallery();
      if (attachment != null) {
        setAttachment(attachment);
      }
    } catch (e) {
      _appendSystemMessage('Sistema: Erro ao selecionar imagem ($e)');
    }
  }

  Future<void> pickAudio() async {
    try {
      final attachment = await _mediaPickerService.pickAudioFile();
      if (attachment != null) {
        setAttachment(attachment);
      }
    } catch (e) {
      _appendSystemMessage('Sistema: Erro ao selecionar arquivo de áudio ($e)');
    }
  }

  Future<void> pickDocument() async {
    try {
      final attachment = await _mediaPickerService.pickDocumentFile();
      if (attachment != null) {
        setAttachment(attachment);
      }
    } catch (e) {
      _appendSystemMessage('Sistema: Erro ao selecionar documento ($e)');
    }
  }

  Future<void> sendMessage(
    String text, [
    ChatAttachment? customAttachment,
  ]) async {
    final attachment = customAttachment ?? attachedAttachment.value;
    final trimmedText = text.trim();

    String effectiveText = trimmedText;
    if (effectiveText.isEmpty && attachment != null) {
      switch (attachment.type) {
        case AttachmentType.image:
          effectiveText = 'Analise esta imagem';
          break;
        case AttachmentType.audio:
          effectiveText = 'Analise este áudio';
          break;
        case AttachmentType.document:
          effectiveText = 'Analise este documento/código';
          break;
      }
    }

    final currentModel = selectedModelName.value;
    if (effectiveText.isEmpty ||
        currentModel.isEmpty ||
        isGenerating.value) {
      return;
    }

    isModelLoaded.value = true;
    clearAttachment();
    _stopGenerationFlag = false;
    isGenerating.value = true;

    // Derive a short conversation title from the first prompt.
    final conv = currentConversation;
    if (conv != null &&
        (conv.title == 'Nova Conversa' || conv.messages.isEmpty)) {
      conv.title = effectiveText.length > 25
          ? '${effectiveText.substring(0, 25)}...'
          : effectiveText;
    }

    final userMessage = ChatMessage(
      text: effectiveText,
      isUser: true,
      attachment: attachment,
      imageBytes:
          attachment?.type == AttachmentType.image ? attachment?.bytes : null,
    );

    // Append user prompt and an empty assistant placeholder for streaming tokens.
    chatHistory.value = [
      ...chatHistory.value,
      userMessage,
      ChatMessage(text: '', isUser: false),
    ];
    _syncMessagesToCurrentConversation();

    final promptToSend = responseStyle.value.applyToPrompt(effectiveText);

    try {
      final stream = _repository.sendMessage(
        text: promptToSend,
        attachment: attachment,
        imageBytes:
            attachment?.type == AttachmentType.image ? attachment?.bytes : null,
        modelName: currentModel,
        history: chatHistory.value.sublist(0, chatHistory.value.length - 2),
        responseStyle: responseStyle.value,
        onMetrics: (pTokens, eTokens) {
          currentMetrics.value = currentMetrics.value.copyWith(
            promptTokens: pTokens,
            evalTokens: eTokens,
          );
          final c = currentConversation;
          if (c != null) {
            c.lastPromptTokens = pTokens;
            c.lastEvalTokens = eTokens;
          }
        },
      );

      await for (final chunk in stream) {
        if (_stopGenerationFlag) break;

        final currentList = List<ChatMessage>.from(chatHistory.value);
        final lastIndex = currentList.length - 1;
        final lastMsg = currentList[lastIndex];

        currentList[lastIndex] = lastMsg.copyWith(
          thinkingContent: lastMsg.thinkingContent + chunk.thinkingDelta,
          content: lastMsg.content + chunk.contentDelta,
          isThinking: chunk.isDone ? false : chunk.isThinking,
        );
        chatHistory.value = currentList;
      }
    } catch (e) {
      if (!_stopGenerationFlag) {
        final currentList = List<ChatMessage>.from(chatHistory.value);
        final lastIndex = currentList.length - 1;
        currentList[lastIndex] = currentList[lastIndex].copyWith(
          content: 'Sistema: Erro ao gerar resposta ($e)',
          isThinking: false,
        );
        chatHistory.value = currentList;
      }
    } finally {
      final currentList = List<ChatMessage>.from(chatHistory.value);
      if (currentList.isNotEmpty && currentList.last.isThinking) {
        final lastIndex = currentList.length - 1;
        currentList[lastIndex] = currentList[lastIndex].copyWith(
          isThinking: false,
        );
        chatHistory.value = currentList;
      }
      isGenerating.value = false;
      _syncMessagesToCurrentConversation();
      await _saveCurrentState();
    }
  }

  void _syncMessagesToCurrentConversation() {
    final conv = currentConversation;
    if (conv != null) {
      conv.messages = List.from(chatHistory.value);
      conv.updatedAt = DateTime.now();
    }
  }

  Future<void> _saveCurrentState() async {
    await _repository.saveConversations(conversations.value);
  }

  void stopGeneration() {
    if (isGenerating.value) {
      _stopGenerationFlag = true;
      _repository.stopGeneration();
      final currentList = List<ChatMessage>.from(chatHistory.value);
      if (currentList.isNotEmpty && currentList.last.isThinking) {
        final lastIndex = currentList.length - 1;
        currentList[lastIndex] = currentList[lastIndex].copyWith(
          isThinking: false,
        );
        chatHistory.value = currentList;
      }
      isGenerating.value = false;
      _syncMessagesToCurrentConversation();
      _saveCurrentState();
    }
  }

  void reset() {
    if (isGenerating.value) return;
    _repository.reset();
  }

  void dispose() {
    _repository.reset();
  }
}

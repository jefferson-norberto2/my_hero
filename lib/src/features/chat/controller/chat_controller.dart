import 'dart:async';
import 'dart:typed_data';
import 'package:flutter/foundation.dart' show TargetPlatform;
import 'package:flutter_edge_ai/flutter_edge_ai.dart';
import 'package:flutter_edge_ai_litertlm/flutter_edge_ai_litertlm.dart';
import 'package:signals_flutter/signals_flutter.dart';
import '../../../core/constants/app_constants.dart';
import '../../../shared/services/media_picker_service.dart';
import '../data/adapter/chat_message_adapter.dart';
import '../data/adapter/conversation_model.dart';
import '../data/adapter/response_style.dart';
import '../data/repository/chat_repository.dart';

class ChatController {
  final ChatRepository _repository;
  final MediaPickerService _mediaPickerService;
  final TargetPlatform? _platformOverride;

  ChatController({
    required this._repository,
    required this._mediaPickerService,
    TargetPlatform? platform,
  })  : _platformOverride = platform,
        maxTokens = signal<int>(
          AppConstants.defaultMaxTokensForPlatform(platform),
        );

  // Signals de estado reativo
  final isModelLoaded = signal(false);
  final isLoading = signal(false);
  final isGenerating = signal(false);
  final downloadProgress = signal(0.0);
  final loadingStatus = signal('');
  final selectedBackend = signal<PreferredBackend>(PreferredBackend.cpu);
  final activeBackend = signal<PreferredBackend?>(null);
  final is2BModel = signal<bool>(true);
  final Signal<int> maxTokens;
  final enableThinking = signal<bool>(false);
  final responseStyle = signal<ResponseStyle>(ResponseStyle.simple);
  final chatHistory = signal<List<ChatMessage>>([]);
  final errorMessage = signal<String?>(null);
  final attachedImage = signal<Uint8List?>(null);

  // Histórico de conversas
  final conversations = signal<List<Conversation>>([]);
  final currentConversationId = signal<String?>(null);

  bool _stopGenerationFlag = false;

  List<PreferredBackend> get availableBackends =>
      AppConstants.availableBackendsForPlatform(_platformOverride);

  bool get supportsNpu =>
      AppConstants.supportsNpuOnPlatform(_platformOverride);

  bool get supportsCamera =>
      AppConstants.supportsCameraOnPlatform(_platformOverride);

  bool get isCpu => selectedBackend.value == PreferredBackend.cpu;

  GemmaModelSpec get currentModelSpec =>
      AppConstants.modelSpec(is2B: is2BModel.value);

  int get minSupportedContextTokens => currentModelSpec.minContextTokens;

  int get maxSupportedContextTokens => currentModelSpec.maxContextTokens;

  String get backendDisplaySummary {
    final requested = selectedBackend.value;
    final actual = activeBackend.value;
    if (actual == null || actual == requested) {
      return AppConstants.backendLabel(actual ?? requested);
    }
    return '${AppConstants.backendLabel(requested)} → ${AppConstants.backendLabel(actual)}';
  }

  Conversation? get currentConversation {
    final id = currentConversationId.value;
    if (id == null) return null;
    try {
      return conversations.value.firstWhere((c) => c.id == id);
    } catch (_) {
      return null;
    }
  }

  PreferredBackend _normalizeBackendForPlatform(PreferredBackend backend) {
    final available = availableBackends;
    if (available.contains(backend)) return backend;
    if (available.contains(PreferredBackend.gpu)) return PreferredBackend.gpu;
    return PreferredBackend.cpu;
  }

  Future<void> init() async {
    await loadSavedConversations();
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

  /// Cria uma nova conversa com contexto 100% limpo e isolado
  Future<void> createNewConversation() async {
    final now = DateTime.now();
    final newConv = Conversation(
      id: now.millisecondsSinceEpoch.toString(),
      title: 'Nova Conversa',
      messages: [],
      is2B: is2BModel.value,
      preferredBackend: _normalizeBackendForPlatform(selectedBackend.value),
      maxTokens: maxTokens.value,
      enableThinking: enableThinking.value,
      responseStyle: responseStyle.value,
      createdAt: now,
      updatedAt: now,
    );

    conversations.value = [newConv, ...conversations.value];
    currentConversationId.value = newConv.id;
    chatHistory.value = [];

    // Se o modelo já estiver carregado, inicia uma nova sessão de chat no Gemma
    // sem nenhum histórico, garantindo que o modelo não misture o contexto anterior!
    if (isModelLoaded.value) {
      try {
        await _repository.startNewChatSession(
          contextMessages: null,
          responseStyle: responseStyle.value,
          enableThinking: enableThinking.value,
        );
      } catch (_) {}
    }

    await _saveCurrentState();
  }

  /// Seleciona uma conversa e carrega exclusivamente o contexto e estilo dela
  Future<void> selectConversation(String id) async {
    if (isGenerating.value) return;

    final conv = conversations.value.where((c) => c.id == id).firstOrNull;
    if (conv == null) return;

    final normalizedBackend =
        _normalizeBackendForPlatform(conv.preferredBackend);
    final clampedTokens = AppConstants.clampMaxTokens(
      conv.maxTokens,
      is2B: conv.is2B,
    );

    currentConversationId.value = conv.id;
    chatHistory.value = List.from(conv.messages);
    is2BModel.value = conv.is2B;
    selectedBackend.value = normalizedBackend;
    maxTokens.value = clampedTokens;
    enableThinking.value = conv.enableThinking;
    responseStyle.value = conv.responseStyle;

    // Se o modelo estiver carregado, reinicia a sessão de chat e sincroniza
    // APENAS as mensagens da conversa selecionada com o estilo configurado
    if (isModelLoaded.value) {
      try {
        await _repository.startNewChatSession(
          contextMessages: conv.messages,
          responseStyle: conv.responseStyle,
          enableThinking: conv.enableThinking,
        );
      } catch (_) {}
    }
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

  void selectBackend(PreferredBackend backend) {
    if (isLoading.value) return;
    final normalized = _normalizeBackendForPlatform(backend);
    selectedBackend.value = normalized;

    final conv = currentConversation;
    if (conv != null) {
      conv.preferredBackend = normalized;
      _saveCurrentState();
    }
  }

  void setMaxTokens(int value) {
    if (isLoading.value) return;
    final clamped = AppConstants.clampMaxTokens(
      value,
      is2B: is2BModel.value,
    );
    maxTokens.value = clamped;

    final conv = currentConversation;
    if (conv != null) {
      conv.maxTokens = clamped;
      _saveCurrentState();
    }
  }

  void setEnableThinking(bool enabled) {
    if (isLoading.value) return;
    enableThinking.value = enabled;

    final conv = currentConversation;
    if (conv != null) {
      conv.enableThinking = enabled;
      _saveCurrentState();
    }
  }

  void cancelDownload() {
    if (!isLoading.value) return;
    loadingStatus.value = 'Cancelando download...';
    _repository.cancelDownload();
  }

  Future<void> downloadAndLoadModel({
    bool is2B = true,
    List<ChatMessage>? contextMessages,
    ResponseStyle? responseStyle,
    PreferredBackend? preferredBackend,
    int? maxTokens,
    bool? enableThinking,
  }) async {
    isLoading.value = true;
    downloadProgress.value = 0.0;
    loadingStatus.value = 'Verificando armazenamento local...';
    errorMessage.value = null;

    final spec = AppConstants.modelSpec(is2B: is2B);
    final targetBackend = _normalizeBackendForPlatform(
      preferredBackend ?? selectedBackend.value,
    );
    final targetMaxTokens = AppConstants.clampMaxTokens(
      maxTokens ?? this.maxTokens.value,
      is2B: is2B,
    );
    final targetThinking = enableThinking ?? this.enableThinking.value;
    final targetStyle = responseStyle ?? this.responseStyle.value;

    try {
      final messagesToSync = contextMessages ?? chatHistory.value;

      await _repository.loadModel(
        is2B: is2B,
        preferredBackend: targetBackend,
        maxTokens: targetMaxTokens,
        onProgress: (progress, status) {
          downloadProgress.value = progress;
          loadingStatus.value = status;
        },
        contextMessages: messagesToSync,
        responseStyle: targetStyle,
        enableThinking: targetThinking,
      );

      final actualBackend = _repository.activeBackend ?? targetBackend;

      is2BModel.value = is2B;
      selectedBackend.value = targetBackend;
      activeBackend.value = actualBackend;
      this.maxTokens.value = targetMaxTokens;
      this.enableThinking.value = targetThinking;
      this.responseStyle.value = targetStyle;
      isModelLoaded.value = true;
      isLoading.value = false;

      final conv = currentConversation;
      if (conv != null) {
        conv.is2B = is2B;
        conv.preferredBackend = targetBackend;
        conv.maxTokens = targetMaxTokens;
        conv.enableThinking = targetThinking;
        conv.responseStyle = targetStyle;
        _saveCurrentState();
      }

      final backendNote = actualBackend == targetBackend
          ? AppConstants.backendLabel(actualBackend)
          : '${AppConstants.backendLabel(targetBackend)} → ${AppConstants.backendLabel(actualBackend)} (fallback)';
      final thinkingNote = targetThinking ? ' • Thinking ativo' : '';
      final contextNote = messagesToSync.isNotEmpty
          ? ' Contexto sincronizado'
          : '';

      chatHistory.value = [
        ...chatHistory.value,
        ChatMessage(
          text:
              'Sistema: Modelo ${spec.displayName} carregado ($backendNote • ${AppConstants.formatTokenCount(targetMaxTokens)} tokens$thinkingNote • ${targetStyle.label}).$contextNote',
          isUser: false,
        ),
      ];
      _syncMessagesToCurrentConversation();
      await _saveCurrentState();
    } on DownloadCancelledException {
      isLoading.value = false;
      loadingStatus.value = '';
      errorMessage.value = 'Download cancelado pelo usuário.';
    } on ModelValidationException catch (e) {
      isLoading.value = false;
      loadingStatus.value = '';
      errorMessage.value =
          'O arquivo baixado falhou na verificação de integridade (${e.message}). Tente baixar novamente.';
    } on LiteRtLmBundleException catch (e) {
      isLoading.value = false;
      loadingStatus.value = '';
      errorMessage.value = switch (e.problem) {
        LiteRtLmBundleProblem.incomplete =>
          'O arquivo do modelo está incompleto ou corrompido. Tente baixar novamente.',
        LiteRtLmBundleProblem.wrongBackend ||
        LiteRtLmBundleProblem.unsupportedNpu =>
          'O modelo selecionado não é compatível com o backend escolhido: ${e.message}',
        _ => 'Arquivo de modelo inválido ou incompatível: ${e.message}',
      };
    } on NoActiveModelError {
      isLoading.value = false;
      loadingStatus.value = '';
      errorMessage.value =
          'Nenhum modelo ativo foi encontrado. Inicie o download novamente.';
    } on BackendInitException catch (e) {
      isLoading.value = false;
      loadingStatus.value = '';
      final backendName = AppConstants.backendLabel(e.lastAttempt.backend);
      errorMessage.value =
          'Falha ao inicializar o modelo em $backendName: ${e.lastAttempt.error}';
    } catch (e) {
      isLoading.value = false;
      errorMessage.value = 'Falha ao carregar o modelo: $e';
    }
  }

  /// Altera as configurações da conversa (modelo, backend, maxTokens, thinking e estilo)
  Future<void> updateConversationSettings({
    required bool is2B,
    required PreferredBackend preferredBackend,
    required int maxTokens,
    required bool enableThinking,
    required ResponseStyle newResponseStyle,
  }) async {
    if (isLoading.value || isGenerating.value) return;

    final normalizedBackend = _normalizeBackendForPlatform(preferredBackend);
    final clampedTokens = AppConstants.clampMaxTokens(maxTokens, is2B: is2B);

    final requiresModelReload = (is2B != is2BModel.value) ||
        (normalizedBackend != selectedBackend.value) ||
        (clampedTokens != this.maxTokens.value);
    final requiresSessionRestart = (newResponseStyle != responseStyle.value) ||
        (enableThinking != this.enableThinking.value);

    if (!requiresModelReload && !requiresSessionRestart) return;

    responseStyle.value = newResponseStyle;
    selectedBackend.value = normalizedBackend;
    this.maxTokens.value = clampedTokens;
    this.enableThinking.value = enableThinking;

    final conv = currentConversation;
    if (conv != null) {
      conv.is2B = is2B;
      conv.preferredBackend = normalizedBackend;
      conv.maxTokens = clampedTokens;
      conv.enableThinking = enableThinking;
      conv.responseStyle = newResponseStyle;
      await _saveCurrentState();
    }

    if (requiresModelReload) {
      await downloadAndLoadModel(
        is2B: is2B,
        preferredBackend: normalizedBackend,
        maxTokens: clampedTokens,
        enableThinking: enableThinking,
        contextMessages: chatHistory.value,
        responseStyle: newResponseStyle,
      );
    } else if (requiresSessionRestart && isModelLoaded.value) {
      await _repository.startNewChatSession(
        contextMessages: chatHistory.value,
        responseStyle: newResponseStyle,
        enableThinking: enableThinking,
      );
      final thinkingStatus = enableThinking ? 'ativado' : 'desativado';
      chatHistory.value = [
        ...chatHistory.value,
        ChatMessage(
          text:
              'Sistema: Configurações atualizadas (Estilo: "${newResponseStyle.label}" • Thinking $thinkingStatus).',
          isUser: false,
        ),
      ];
      _syncMessagesToCurrentConversation();
      await _saveCurrentState();
    }
  }

  /// Mantido para compatibilidade
  Future<void> switchModelWithContext({
    required bool is2B,
    required PreferredBackend preferredBackend,
  }) async {
    await updateConversationSettings(
      is2B: is2B,
      preferredBackend: preferredBackend,
      maxTokens: maxTokens.value,
      enableThinking: enableThinking.value,
      newResponseStyle: responseStyle.value,
    );
  }

  void setAttachedImage(Uint8List? bytes) {
    attachedImage.value = bytes;
  }

  void clearAttachedImage() {
    attachedImage.value = null;
  }

  Future<void> pickImage({bool fromCamera = false}) async {
    try {
      final useCamera = fromCamera && supportsCamera;
      final imageBytes = useCamera
          ? await _mediaPickerService.pickImageFromCamera()
          : await _mediaPickerService.pickImageFromGallery();
      if (imageBytes != null) {
        attachedImage.value = imageBytes;
      }
    } catch (e) {
      chatHistory.value = [
        ...chatHistory.value,
        ChatMessage(
          text: 'Sistema: Erro ao selecionar imagem ($e)',
          isUser: false,
        ),
      ];
      _syncMessagesToCurrentConversation();
    }
  }

  Future<void> sendMessage(String text, [Uint8List? imageBytes]) async {
    final image = imageBytes ?? attachedImage.value;
    final trimmedText = text.trim();
    final effectiveText = trimmedText.isEmpty && image != null
        ? 'Analise esta imagem'
        : trimmedText;

    if (effectiveText.isEmpty || !isModelLoaded.value || isGenerating.value) {
      return;
    }

    attachedImage.value = null;
    _stopGenerationFlag = false;
    isGenerating.value = true;

    // Atualiza o título se for uma conversa nova
    final conv = currentConversation;
    if (conv != null &&
        (conv.title == 'Nova Conversa' || conv.messages.isEmpty)) {
      conv.title = effectiveText.length > 25
          ? '${effectiveText.substring(0, 25)}...'
          : effectiveText;
    }

    // Adiciona a pergunta do usuário e uma resposta vazia para preenchimento em stream
    chatHistory.value = [
      ...chatHistory.value,
      ChatMessage(text: effectiveText, isUser: true, imageBytes: image),
      ChatMessage(text: '', isUser: false),
    ];
    _syncMessagesToCurrentConversation();

    final promptToSend = responseStyle.value.applyToPrompt(effectiveText);

    try {
      final stream = _repository.sendMessage(
        text: promptToSend,
        imageBytes: image,
      );

      await for (final chunk in stream) {
        if (_stopGenerationFlag) break;

        final currentList = List<ChatMessage>.from(chatHistory.value);
        final lastIndex = currentList.length - 1;
        final lastMsg = currentList[lastIndex];

        switch (chunk) {
          case ChatTextChunk(:final token):
            currentList[lastIndex] = lastMsg.copyWith(
              text: lastMsg.text + token,
            );
          case ChatThinkingChunk(:final content):
            currentList[lastIndex] = lastMsg.copyWith(
              thinkingText: (lastMsg.thinkingText ?? '') + content,
            );
        }
        chatHistory.value = currentList;
      }
    } on ContextLimitException {
      if (!_stopGenerationFlag) {
        final currentList = List<ChatMessage>.from(chatHistory.value);
        final lastIndex = currentList.length - 1;
        currentList[lastIndex] = currentList[lastIndex].copyWith(
          text:
              'Sistema: A janela de contexto (${AppConstants.formatTokenCount(maxTokens.value)} tokens) foi atingida. Encurte a mensagem, aumente o limite de tokens nas configurações ou inicie uma nova conversa.',
        );
        chatHistory.value = currentList;
      }
    } catch (e) {
      if (!_stopGenerationFlag) {
        final currentList = List<ChatMessage>.from(chatHistory.value);
        final lastIndex = currentList.length - 1;
        currentList[lastIndex] = currentList[lastIndex].copyWith(
          text: 'Sistema: Erro ao gerar resposta ($e)',
        );
        chatHistory.value = currentList;
      }
    } finally {
      isGenerating.value = false;
      _syncMessagesToCurrentConversation();
      await _saveCurrentState();
    }
  }

  Future<void> pickAndSendImage(
    String currentText, {
    bool fromCamera = false,
  }) async {
    try {
      final useCamera = fromCamera && supportsCamera;
      final imageBytes = useCamera
          ? await _mediaPickerService.pickImageFromCamera()
          : await _mediaPickerService.pickImageFromGallery();
      if (imageBytes != null) {
        await sendMessage(
          currentText.isEmpty ? 'Analise esta imagem' : currentText,
          imageBytes,
        );
      }
    } catch (e) {
      chatHistory.value = [
        ...chatHistory.value,
        ChatMessage(
          text: 'Sistema: Erro ao selecionar imagem ($e)',
          isUser: false,
        ),
      ];
      _syncMessagesToCurrentConversation();
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

  Future<void> stopGeneration() async {
    if (isGenerating.value) {
      _stopGenerationFlag = true;
      await _repository.stopGeneration();
      isGenerating.value = false;
      _syncMessagesToCurrentConversation();
      await _saveCurrentState();
    }
  }

  Future<void> reset() async {
    if (isGenerating.value) return;
    await _repository.reset();
    isModelLoaded.value = false;
    activeBackend.value = null;
  }

  void dispose() {
    unawaited(_repository.reset());
  }
}

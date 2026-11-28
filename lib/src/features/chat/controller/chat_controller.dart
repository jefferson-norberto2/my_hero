import 'dart:async';
import 'dart:typed_data';
import 'package:flutter/foundation.dart' show TargetPlatform;
import 'package:flutter_edge_ai/flutter_edge_ai.dart';
import 'package:flutter_edge_ai_litertlm/flutter_edge_ai_litertlm.dart';
import 'package:signals_flutter/signals_flutter.dart';
import 'package:my_hero/src/core/constants/app_constants.dart';
import 'package:my_hero/src/shared/services/inference_notification_service.dart';
import 'package:my_hero/src/shared/services/media_picker_service.dart';
import 'package:my_hero/src/features/chat/data/adapter/chat_message_adapter.dart';
import 'package:my_hero/src/features/chat/data/adapter/conversation_model.dart';
import 'package:my_hero/src/features/chat/data/adapter/response_style.dart';
import 'package:my_hero/src/features/chat/data/repository/chat_repository.dart';

class ChatController {
  final ChatRepository _repository;
  final MediaPickerService _mediaPickerService;
  final InferenceNotificationService _inferenceNotificationService;
  final TargetPlatform? _platformOverride;

  ChatController({
    required this._repository,
    required this._mediaPickerService,
    InferenceNotificationService? inferenceNotificationService,
    TargetPlatform? platform,
  })  : _inferenceNotificationService =
            inferenceNotificationService ??
            const NoopInferenceNotificationService(),
        _platformOverride = platform,
        maxTokens = signal<int>(
          AppConstants.defaultMaxTokensForPlatform(platform),
        ) {
    _inferenceNotificationService.setOnStopRequested(stopGeneration);
  }

  // Signals de estado reativo
  final isModelLoaded = signal(false);
  final isLoading = signal(false);
  final isGenerating = signal(false);
  final downloadProgress = signal(0.0);
  final loadingStatus = signal('');
  final selectedModelId = signal<String>(AppConstants.defaultModelId);
  final loadedModelId = signal<String?>(null);
  final installedModelFiles = signal<Set<String>>({});
  final deletingModelFileName = signal<String?>(null);
  final selectedBackend = signal<PreferredBackend>(PreferredBackend.cpu);
  final activeBackend = signal<PreferredBackend?>(null);
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

  List<AiModelSpec> get availableModels => AppConstants.availableModels;

  List<PreferredBackend> get availableBackends =>
      AppConstants.availableBackendsForPlatform(_platformOverride);

  bool get supportsNpu =>
      AppConstants.supportsNpuOnPlatform(_platformOverride);

  bool get supportsCamera =>
      AppConstants.supportsCameraOnPlatform(_platformOverride);

  bool get isCpu => selectedBackend.value == PreferredBackend.cpu;

  AiModelSpec get currentModelSpec =>
      AppConstants.modelSpecById(selectedModelId.value);

  AiModelSpec get activeLoadedModelSpec =>
      AppConstants.modelSpecById(loadedModelId.value ?? selectedModelId.value);

  bool get supportsImage =>
      (isModelLoaded.value ? activeLoadedModelSpec : currentModelSpec)
          .supportsImage;

  bool get supportsThinking => currentModelSpec.supportsThinking;

  bool isModelInstalled(AiModelSpec spec) =>
      installedModelFiles.value.contains(spec.fileName);

  int get installedModelsCount => availableModels
      .where((spec) => installedModelFiles.value.contains(spec.fileName))
      .length;

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
    await Future.wait([
      loadSavedConversations(),
      refreshInstalledModels(),
    ]);
  }

  Future<void> refreshInstalledModels() async {
    try {
      final installed = await _repository.getInstalledModelFileNames();
      installedModelFiles.value = Set<String>.unmodifiable(installed);
    } catch (_) {
      // Falha silenciosa para não bloquear a inicialização caso o storage não esteja pronto
    }
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
    final spec = currentModelSpec;
    final effectiveThinking = enableThinking.value && spec.supportsThinking;
    final newConv = Conversation(
      id: now.millisecondsSinceEpoch.toString(),
      title: 'Nova Conversa',
      messages: [],
      modelId: spec.id,
      preferredBackend: _normalizeBackendForPlatform(selectedBackend.value),
      maxTokens: maxTokens.value,
      enableThinking: effectiveThinking,
      responseStyle: responseStyle.value,
      createdAt: now,
      updatedAt: now,
    );

    conversations.value = [newConv, ...conversations.value];
    currentConversationId.value = newConv.id;
    chatHistory.value = [];

    if (isModelLoaded.value) {
      try {
        await _repository.startNewChatSession(
          modelId: loadedModelId.value ?? spec.id,
          contextMessages: null,
          responseStyle: responseStyle.value,
          enableThinking: effectiveThinking,
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

    final spec = AppConstants.modelSpecById(conv.modelId);
    final normalizedBackend =
        _normalizeBackendForPlatform(conv.preferredBackend);
    final clampedTokens = AppConstants.clampMaxTokens(
      conv.maxTokens,
      modelId: spec.id,
    );
    final effectiveThinking = conv.enableThinking && spec.supportsThinking;

    currentConversationId.value = conv.id;
    chatHistory.value = List.from(conv.messages);
    selectedModelId.value = spec.id;
    selectedBackend.value = normalizedBackend;
    maxTokens.value = clampedTokens;
    enableThinking.value = effectiveThinking;
    responseStyle.value = conv.responseStyle;

    if (!spec.supportsImage) {
      attachedImage.value = null;
    }

    if (isModelLoaded.value) {
      try {
        await _repository.startNewChatSession(
          modelId: loadedModelId.value ?? spec.id,
          contextMessages: conv.messages,
          responseStyle: conv.responseStyle,
          enableThinking: effectiveThinking,
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

  void selectModel(String modelId) {
    if (isLoading.value) return;
    final spec = AppConstants.modelSpecById(modelId);
    final clampedTokens = AppConstants.clampMaxTokens(
      maxTokens.value,
      modelId: spec.id,
    );
    final effectiveThinking = enableThinking.value && spec.supportsThinking;

    selectedModelId.value = spec.id;
    maxTokens.value = clampedTokens;
    enableThinking.value = effectiveThinking;
    if (!spec.supportsImage) {
      attachedImage.value = null;
    }

    final conv = currentConversation;
    if (conv != null) {
      conv.modelId = spec.id;
      conv.maxTokens = clampedTokens;
      conv.enableThinking = effectiveThinking;
      _saveCurrentState();
    }
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
      modelId: selectedModelId.value,
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
    final effective = enabled && currentModelSpec.supportsThinking;
    enableThinking.value = effective;

    final conv = currentConversation;
    if (conv != null) {
      conv.enableThinking = effective;
      _saveCurrentState();
    }
  }

  void cancelDownload() {
    if (!isLoading.value) return;
    loadingStatus.value = 'Cancelando download...';
    _repository.cancelDownload();
  }

  /// Exclui um modelo já baixado do armazenamento local para liberar espaço.
  Future<void> uninstallModel(AiModelSpec spec) async {
    if (isLoading.value ||
        isGenerating.value ||
        deletingModelFileName.value != null) {
      return;
    }

    deletingModelFileName.value = spec.fileName;
    errorMessage.value = null;

    try {
      final wasLoaded = isModelLoaded.value &&
          (loadedModelId.value == spec.id ||
              (loadedModelId.value == null &&
                  selectedModelId.value == spec.id));

      await _repository.uninstallModel(spec.fileName);

      if (wasLoaded) {
        isModelLoaded.value = false;
        loadedModelId.value = null;
        activeBackend.value = null;
      }

      await refreshInstalledModels();
      // Garante remoção otimista caso o mock/repositório não retorne nova lista
      if (installedModelFiles.value.contains(spec.fileName)) {
        final next = Set<String>.from(installedModelFiles.value)
          ..remove(spec.fileName);
        installedModelFiles.value = Set<String>.unmodifiable(next);
      }
    } catch (e) {
      errorMessage.value = 'Falha ao excluir ${spec.displayName}: $e';
    } finally {
      deletingModelFileName.value = null;
    }
  }

  Future<void> downloadAndLoadModel({
    String? modelId,
    bool? is2B,
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

    final resolvedModelId = modelId ??
        (is2B != null
            ? AppConstants.modelSpec(is2B: is2B).id
            : selectedModelId.value);
    final spec = AppConstants.modelSpecById(resolvedModelId);
    final targetBackend = _normalizeBackendForPlatform(
      preferredBackend ?? selectedBackend.value,
    );
    final targetMaxTokens = AppConstants.clampMaxTokens(
      maxTokens ?? this.maxTokens.value,
      modelId: spec.id,
    );
    final targetThinking =
        (enableThinking ?? this.enableThinking.value) && spec.supportsThinking;
    final targetStyle = responseStyle ?? this.responseStyle.value;

    try {
      final messagesToSync = contextMessages ?? chatHistory.value;

      await _repository.loadModel(
        modelId: spec.id,
        is2B: spec.id == AppConstants.model2b.id,
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

      selectedModelId.value = spec.id;
      loadedModelId.value = spec.id;
      selectedBackend.value = targetBackend;
      activeBackend.value = actualBackend;
      this.maxTokens.value = targetMaxTokens;
      this.enableThinking.value = targetThinking;
      this.responseStyle.value = targetStyle;
      if (!spec.supportsImage) {
        attachedImage.value = null;
      }
      isModelLoaded.value = true;
      isLoading.value = false;

      // Marca como instalado imediatamente e sincroniza com o armazenamento
      installedModelFiles.value = Set<String>.unmodifiable({
        ...installedModelFiles.value,
        spec.fileName,
      });
      unawaited(refreshInstalledModels());

      final conv = currentConversation;
      if (conv != null) {
        conv.modelId = spec.id;
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
    } on DownloadException catch (e) {
      isLoading.value = false;
      loadingStatus.value = '';
      errorMessage.value = switch (e.error) {
        CanceledError() => 'Download cancelado pelo usuário.',
        NetworkError(:final message) =>
          'Erro de conexão durante o download ($message). Verifique sua internet e tente novamente.',
        RateLimitedError() =>
          'Limite de requisições atingido no servidor (HTTP 429). Aguarde alguns instantes e tente novamente.',
        ServerError(:final statusCode) =>
          'Erro temporário no servidor de download (HTTP $statusCode). Tente novamente mais tarde.',
        UnauthorizedError() || ForbiddenError() =>
          'Acesso negado ao arquivo do modelo no servidor.',
        NotFoundError() =>
          'Arquivo do modelo não encontrado no servidor (HTTP 404).',
        UnknownError(:final message) => 'Falha no download do modelo: $message',
      };
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
    String? modelId,
    bool? is2B,
    required PreferredBackend preferredBackend,
    required int maxTokens,
    required bool enableThinking,
    required ResponseStyle newResponseStyle,
  }) async {
    if (isLoading.value || isGenerating.value) return;

    final resolvedModelId = modelId ??
        (is2B != null
            ? AppConstants.modelSpec(is2B: is2B).id
            : selectedModelId.value);
    final spec = AppConstants.modelSpecById(resolvedModelId);
    final normalizedBackend = _normalizeBackendForPlatform(preferredBackend);
    final clampedTokens = AppConstants.clampMaxTokens(
      maxTokens,
      modelId: spec.id,
    );
    final effectiveThinking = enableThinking && spec.supportsThinking;

    final currentLoaded = loadedModelId.value ?? selectedModelId.value;
    final requiresModelReload = !isModelLoaded.value ||
        (spec.id != currentLoaded) ||
        (normalizedBackend != selectedBackend.value) ||
        (clampedTokens != this.maxTokens.value);
    final requiresSessionRestart = (newResponseStyle != responseStyle.value) ||
        (effectiveThinking != this.enableThinking.value);

    if (!requiresModelReload && !requiresSessionRestart) return;

    responseStyle.value = newResponseStyle;
    selectedModelId.value = spec.id;
    selectedBackend.value = normalizedBackend;
    this.maxTokens.value = clampedTokens;
    this.enableThinking.value = effectiveThinking;
    if (!spec.supportsImage) {
      attachedImage.value = null;
    }

    final conv = currentConversation;
    if (conv != null) {
      conv.modelId = spec.id;
      conv.preferredBackend = normalizedBackend;
      conv.maxTokens = clampedTokens;
      conv.enableThinking = effectiveThinking;
      conv.responseStyle = newResponseStyle;
      await _saveCurrentState();
    }

    if (requiresModelReload) {
      await downloadAndLoadModel(
        modelId: spec.id,
        preferredBackend: normalizedBackend,
        maxTokens: clampedTokens,
        enableThinking: effectiveThinking,
        contextMessages: chatHistory.value,
        responseStyle: newResponseStyle,
      );
    } else if (requiresSessionRestart && isModelLoaded.value) {
      await _repository.startNewChatSession(
        modelId: spec.id,
        contextMessages: chatHistory.value,
        responseStyle: newResponseStyle,
        enableThinking: effectiveThinking,
      );
      final thinkingNote = spec.supportsThinking
          ? ' • Thinking ${effectiveThinking ? 'ativado' : 'desativado'}'
          : '';
      chatHistory.value = [
        ...chatHistory.value,
        ChatMessage(
          text:
              'Sistema: Configurações atualizadas (Estilo: "${newResponseStyle.label}"$thinkingNote).',
          isUser: false,
        ),
      ];
      _syncMessagesToCurrentConversation();
      await _saveCurrentState();
    }
  }

  /// Mantido para compatibilidade
  Future<void> switchModelWithContext({
    String? modelId,
    bool? is2B,
    required PreferredBackend preferredBackend,
  }) async {
    await updateConversationSettings(
      modelId: modelId,
      is2B: is2B,
      preferredBackend: preferredBackend,
      maxTokens: maxTokens.value,
      enableThinking: enableThinking.value,
      newResponseStyle: responseStyle.value,
    );
  }

  void setAttachedImage(Uint8List? bytes) {
    if (bytes != null && !supportsImage) return;
    attachedImage.value = bytes;
  }

  void clearAttachedImage() {
    attachedImage.value = null;
  }

  Future<void> pickImage({bool fromCamera = false}) async {
    if (!supportsImage) return;
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
    final activeSpec = activeLoadedModelSpec;
    final image = activeSpec.supportsImage
        ? (imageBytes ?? attachedImage.value)
        : null;
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
    final notificationTitle = 'My Hero • ${activeSpec.displayName}';
    final startsThinking = enableThinking.value && activeSpec.supportsThinking;
    var currentNotificationBody = startsThinking
        ? 'Pensando na resposta...'
        : 'Gerando resposta...';
    var hasError = false;

    await _inferenceNotificationService.start(
      title: notificationTitle,
      body: currentNotificationBody,
    );

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
            if (currentNotificationBody != 'Gerando resposta...') {
              currentNotificationBody = 'Gerando resposta...';
              unawaited(
                _inferenceNotificationService.update(
                  title: notificationTitle,
                  body: currentNotificationBody,
                ),
              );
            }
            currentList[lastIndex] = lastMsg.copyWith(
              text: lastMsg.text + token,
            );
          case ChatThinkingChunk(:final content):
            if (currentNotificationBody != 'Pensando na resposta...') {
              currentNotificationBody = 'Pensando na resposta...';
              unawaited(
                _inferenceNotificationService.update(
                  title: notificationTitle,
                  body: currentNotificationBody,
                ),
              );
            }
            currentList[lastIndex] = lastMsg.copyWith(
              thinkingText: (lastMsg.thinkingText ?? '') + content,
            );
        }
        chatHistory.value = currentList;
      }
    } on ContextLimitException {
      hasError = true;
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
      hasError = true;
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

      final finalAnswer = chatHistory.value.isNotEmpty
          ? chatHistory.value.last.text.trim()
          : '';
      final shouldNotifyCompleted =
          !hasError && !_stopGenerationFlag && finalAnswer.isNotEmpty;

      await _inferenceNotificationService.stop(
        notifyIfBackgrounded: shouldNotifyCompleted,
        completedTitle: shouldNotifyCompleted
            ? 'Resposta pronta (${activeSpec.displayName})'
            : null,
        completedBody: shouldNotifyCompleted
            ? _buildNotificationPreview(finalAnswer)
            : null,
      );
    }
  }

  String _buildNotificationPreview(String text) {
    final normalized = text.replaceAll(RegExp(r'\s+'), ' ').trim();
    if (normalized.length <= 120) return normalized;
    return '${normalized.substring(0, 117)}...';
  }

  Future<void> pickAndSendImage(
    String currentText, {
    bool fromCamera = false,
  }) async {
    if (!supportsImage) return;
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
      await _inferenceNotificationService.stop();
      isGenerating.value = false;
      _syncMessagesToCurrentConversation();
      await _saveCurrentState();
    }
  }

  Future<void> reset() async {
    if (isGenerating.value) return;
    await _repository.reset();
    isModelLoaded.value = false;
    loadedModelId.value = null;
    activeBackend.value = null;
  }

  void dispose() {
    _inferenceNotificationService.setOnStopRequested(null);
    unawaited(_inferenceNotificationService.stop());
    unawaited(_repository.reset());
  }
}

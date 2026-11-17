import 'package:flutter/material.dart';
import 'package:get_it/get_it.dart';
import 'package:go_router/go_router.dart';
import 'package:my_hero/src/core/routes/app_routes.dart';
import 'package:my_hero/src/core/theme/app_theme.dart';
import 'package:my_hero/src/features/chat/controller/chat_controller.dart';
import 'package:my_hero/src/features/chat/data/adapter/chat_metrics.dart';
import 'package:my_hero/src/features/chat/ui/widgets/chat_drawer.dart';
import 'package:my_hero/src/features/chat/ui/widgets/chat_input_bar.dart';
import 'package:my_hero/src/features/chat/ui/widgets/chat_message_list.dart';
import 'package:my_hero/src/features/chat/ui/widgets/chat_metrics_badge.dart';
import 'package:my_hero/src/features/chat/ui/widgets/chat_selection_appbar.dart';
import 'package:my_hero/src/features/settings/data/models/settings_args.dart';
import 'package:my_hero/src/shared/models/compute_mode.dart';
import 'package:my_hero/src/shared/models/context_settings.dart';
import 'package:signals_flutter/signals_flutter.dart';

enum _ChatExportAction {
  exportFull,
  selectMessages,
}

/// Main chat page displaying conversation messages, input bar, and status metrics.
class ChatPage extends StatefulWidget {
  final ChatController? controller;

  const ChatPage({super.key, this.controller});

  @override
  State<ChatPage> createState() => _ChatPageState();
}

class _ChatPageState extends State<ChatPage> {
  late final ChatController _controller;
  EffectCleanup? _errorEffectCleanup;

  @override
  void initState() {
    super.initState();
    _controller = widget.controller ?? GetIt.I<ChatController>();
    _controller.init();

    // Display a floating snackbar whenever an error message is emitted.
    _errorEffectCleanup = effect(() {
      final error = _controller.errorMessage.value;
      if (error != null && mounted) {
        WidgetsBinding.instance.addPostFrameCallback((_) {
          if (!mounted) return;
          ScaffoldMessenger.of(context).showSnackBar(
            SnackBar(
              content: Text(error),
              backgroundColor: Theme.of(context).colorScheme.error,
              behavior: SnackBarBehavior.floating,
            ),
          );
        });
      }
    });
  }

  @override
  void dispose() {
    _errorEffectCleanup?.call();
    _controller.dispose();
    super.dispose();
  }

  Future<void> _openSettings(BuildContext context) async {
    final result = await context.push<SettingsResult>(
      AppRoutes.settings,
      extra: _controller.buildSettingsArgs(),
    );
    if (result != null && mounted) {
      await _controller.applySettings(result);
    }
  }

  String _memoryCardTitle(ComputeMode mode) {
    switch (mode) {
      case ComputeMode.gpu:
        return 'Memória de Vídeo (VRAM)';
      case ComputeMode.cpu:
        return 'Memória Principal (RAM / CPU)';
      case ComputeMode.hybrid:
        return 'Memória Híbrida (GPU/CPU)';
      case ComputeMode.none:
        return 'Memória do Modelo';
    }
  }

  Widget _buildMemoryDetailsCard(BuildContext context, ChatMetrics metrics) {
    final appColors = AppThemeColors.of(context);

    return Container(
      padding: const EdgeInsets.all(12),
      decoration: BoxDecoration(
        color: appColors.infoSurface,
        borderRadius: BorderRadius.circular(10),
        border: Border.all(color: appColors.infoBorder),
      ),
      child: Row(
        children: [
          Icon(
            Icons.memory_rounded,
            color: appColors.infoIcon,
            size: 28,
          ),
          const SizedBox(width: 12),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  _memoryCardTitle(metrics.computeMode),
                  style: const TextStyle(
                    fontWeight: FontWeight.bold,
                    fontSize: 13,
                  ),
                ),
                Text(
                  'Alocado no Ollama: ${metrics.memoryFormatted}',
                  style: TextStyle(
                    fontSize: 12,
                    color: appColors.infoText,
                  ),
                ),
                Text(
                  metrics.hardwareTooltip,
                  style: TextStyle(
                    fontSize: 11,
                    color: appColors.infoSubtext,
                  ),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }

  void _showMetricsDetailsDialog(BuildContext context) {
    final metrics = _controller.currentMetrics.value;
    final theme = Theme.of(context);
    final appColors = AppThemeColors.of(context);

    showDialog(
      context: context,
      builder: (ctx) {
        final usagePercent = metrics.contextUsagePercent;
        Color progressColor = appColors.contextLow;
        if (usagePercent > 80) {
          progressColor = appColors.contextHigh;
        } else if (usagePercent > 50) {
          progressColor = appColors.contextMedium;
        }

        return AlertDialog(
          title: Row(
            children: [
              Icon(Icons.analytics_outlined, color: appColors.accent),
              const SizedBox(width: 8),
              const Text('Métricas do Modelo'),
            ],
          ),
          content: Column(
            mainAxisSize: MainAxisSize.min,
            crossAxisAlignment: CrossAxisAlignment.stretch,
            children: [
              _buildMemoryDetailsCard(ctx, metrics),
              const SizedBox(height: 14),
              // Context Card
              Container(
                padding: const EdgeInsets.all(12),
                decoration: BoxDecoration(
                  color: theme.colorScheme.surfaceContainerHighest
                      .withValues(alpha: 0.5),
                  borderRadius: BorderRadius.circular(10),
                  border: Border.all(
                    color:
                        theme.colorScheme.outlineVariant.withValues(alpha: 0.5),
                  ),
                ),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Row(
                      mainAxisAlignment: MainAxisAlignment.spaceBetween,
                      children: [
                        const Text(
                          'Uso da Janela de Contexto',
                          style: TextStyle(
                            fontWeight: FontWeight.bold,
                            fontSize: 13,
                          ),
                        ),
                        Text(
                          '${usagePercent.toStringAsFixed(1)}%',
                          style: TextStyle(
                            fontWeight: FontWeight.bold,
                            color: progressColor,
                          ),
                        ),
                      ],
                    ),
                    const SizedBox(height: 8),
                    ClipRRect(
                      borderRadius: BorderRadius.circular(6),
                      child: LinearProgressIndicator(
                        value: (usagePercent / 100).clamp(0.0, 1.0),
                        color: progressColor,
                        backgroundColor:
                            theme.colorScheme.surfaceContainerHighest,
                        minHeight: 8,
                      ),
                    ),
                    const SizedBox(height: 10),
                    Text(
                      '• Tokens utilizados: ${metrics.totalTokens} / ${ContextSettings.formatTokenCount(metrics.contextLimit)}',
                      style: const TextStyle(fontSize: 12),
                    ),
                    Text(
                      '• Tokens restantes: ${metrics.remainingTokens} (${metrics.remainingPercent.toStringAsFixed(1)}%)',
                      style: TextStyle(
                        fontSize: 12,
                        fontWeight: FontWeight.w600,
                        color: progressColor,
                      ),
                    ),
                    Text(
                      '• Máximo do modelo: ${ContextSettings.formatTokenCount(metrics.maxContextLength)} tokens',
                      style: TextStyle(
                        fontSize: 11.5,
                        color: theme.colorScheme.outline,
                      ),
                    ),
                    if (metrics.promptTokens > 0) ...[
                      const Divider(height: 16),
                      Text(
                        'Última geração: Prompt: ${metrics.promptTokens} tok | Resposta: ${metrics.evalTokens} tok',
                        style: TextStyle(
                          fontSize: 11,
                          color: theme.colorScheme.outline,
                        ),
                      ),
                    ],
                  ],
                ),
              ),
            ],
          ),
          actions: [
            FilledButton(
              onPressed: () => Navigator.of(ctx).pop(),
              child: const Text('Fechar'),
            ),
          ],
        );
      },
    );
  }

  Widget _buildMetricsBadge(BuildContext context, {required bool isCompact}) {
    return SignalBuilder(
      builder: (context) {
        final isLoaded = _controller.isModelLoaded.value;
        if (!isLoaded) return const SizedBox.shrink();

        if (isCompact) {
          return IconButton(
            icon: const Icon(Icons.analytics_outlined),
            tooltip: 'Métricas do Modelo',
            onPressed: () => _showMetricsDetailsDialog(context),
          );
        }

        return ChatMetricsBadge(
          metrics: _controller.currentMetrics.value,
          onTap: () => _showMetricsDetailsDialog(context),
        );
      },
    );
  }

  Future<void> _handleExportFullConversation() async {
    final exported = await _controller.exportFullConversation();
    if (exported && mounted) {
      _showExportSuccessSnackBar();
    }
  }

  Future<void> _handleExportSelectedMessages() async {
    final exported = await _controller.exportSelectedMessages();
    if (exported && mounted) {
      _showExportSuccessSnackBar();
    }
  }

  void _showExportSuccessSnackBar() {
    ScaffoldMessenger.of(context).showSnackBar(
      const SnackBar(
        content: Text('Arquivo exportado com sucesso!'),
        behavior: SnackBarBehavior.floating,
        duration: Duration(seconds: 2),
      ),
    );
  }

  Widget _buildExportMenuButton(BuildContext context) {
    return SignalBuilder(
      builder: (context) {
        final isGenerating = _controller.isGenerating.value;
        final isExporting = _controller.isExporting.value;
        // Subscribe to chatHistory so menu availability reflects new messages.
        _controller.chatHistory.value;
        final hasExportable = _controller.hasExportableMessages;

        return PopupMenuButton<_ChatExportAction>(
          icon: const Icon(Icons.file_download_outlined),
          tooltip: 'Exportar',
          constraints: const BoxConstraints(minWidth: 268, maxWidth: 320),
          enabled: !isGenerating && !isExporting,
          onSelected: (action) {
            switch (action) {
              case _ChatExportAction.exportFull:
                _handleExportFullConversation();
                break;
              case _ChatExportAction.selectMessages:
                _controller.enterSelectionMode();
                break;
            }
          },
          itemBuilder: (context) => [
            PopupMenuItem<_ChatExportAction>(
              value: _ChatExportAction.exportFull,
              enabled: hasExportable,
              child: const Row(
                mainAxisSize: MainAxisSize.min,
                children: [
                  Icon(Icons.description_outlined, size: 18),
                  SizedBox(width: 8),
                  Flexible(
                    child: Text(
                      'Exportar conversa completa',
                      overflow: TextOverflow.ellipsis,
                    ),
                  ),
                ],
              ),
            ),
            PopupMenuItem<_ChatExportAction>(
              value: _ChatExportAction.selectMessages,
              enabled: hasExportable,
              child: const Row(
                mainAxisSize: MainAxisSize.min,
                children: [
                  Icon(Icons.checklist_rounded, size: 18),
                  SizedBox(width: 8),
                  Flexible(
                    child: Text(
                      'Selecionar mensagens',
                      overflow: TextOverflow.ellipsis,
                    ),
                  ),
                ],
              ),
            ),
          ],
        );
      },
    );
  }

  PreferredSizeWidget _buildDefaultAppBar(
    BuildContext context, {
    required bool isCompact,
  }) {
    return AppBar(
      title: SignalBuilder(
        builder: (context) {
          final conv = _controller.currentConversation;
          final isLoaded = _controller.isModelLoaded.value;
          final isConnected = _controller.isOllamaConnected.value;
          final modelName = _controller.selectedModelName.value;
          final host = _controller.ollamaBaseUrl.value;

          String subtitle;
          if (!isConnected) {
            subtitle = 'Ollama desconectado ($host)';
          } else if (isLoaded && modelName.isNotEmpty) {
            subtitle = '$modelName • ${_controller.responseStyle.value.label}';
          } else if (modelName.isNotEmpty) {
            subtitle = '$modelName (não carregado na memória)';
          } else {
            subtitle = 'Nenhum modelo selecionado ($host)';
          }

          return Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            mainAxisSize: MainAxisSize.min,
            children: [
              Text(
                conv?.title ?? 'Ollama Chat',
                style: const TextStyle(
                  fontSize: 16,
                  fontWeight: FontWeight.bold,
                ),
                overflow: TextOverflow.ellipsis,
              ),
              Text(
                subtitle,
                style: const TextStyle(fontSize: 11),
                overflow: TextOverflow.ellipsis,
              ),
            ],
          );
        },
      ),
      actions: [
        _buildMetricsBadge(context, isCompact: isCompact),
        _buildExportMenuButton(context),
        SignalBuilder(
          builder: (context) {
            final isGenerating = _controller.isGenerating.value;

            return IconButton(
              icon: const Icon(Icons.tune),
              tooltip: 'Configurações',
              onPressed: isGenerating ? null : () => _openSettings(context),
            );
          },
        ),
      ],
    );
  }

  @override
  Widget build(BuildContext context) {
    final isCompact = MediaQuery.sizeOf(context).width < 480;

    return SignalBuilder(
      builder: (context) {
        final isSelectionMode = _controller.isSelectionMode.value;

        return PopScope(
          canPop: !isSelectionMode,
          onPopInvokedWithResult: (didPop, _) {
            if (!didPop && isSelectionMode) {
              _controller.exitSelectionMode();
            }
          },
          child: Scaffold(
            drawer: isSelectionMode
                ? null
                : ChatDrawer(
                    controller: _controller,
                    onOpenSettings: () => _openSettings(context),
                  ),
            drawerEnableOpenDragGesture: !isSelectionMode,
            appBar: isSelectionMode
                ? _ReactiveSelectionAppBar(
                    controller: _controller,
                    isCompact: isCompact,
                    onExport: _handleExportSelectedMessages,
                  )
                : _buildDefaultAppBar(context, isCompact: isCompact),
            body: Padding(
              padding: const EdgeInsets.all(16.0),
              child: Column(
                children: [
                  Expanded(
                    child: ChatMessageList(controller: _controller),
                  ),
                  SignalBuilder(
                    builder: (context) {
                      if (_controller.isLoading.value ||
                          _controller.isGenerating.value ||
                          _controller.isExporting.value) {
                        return const Padding(
                          padding: EdgeInsets.symmetric(vertical: 8.0),
                          child: LinearProgressIndicator(),
                        );
                      }
                      return const SizedBox.shrink();
                    },
                  ),
                  if (!isSelectionMode) ChatInputBar(controller: _controller),
                ],
              ),
            ),
          ),
        );
      },
    );
  }
}

class _ReactiveSelectionAppBar extends StatelessWidget
    implements PreferredSizeWidget {
  final ChatController controller;
  final bool isCompact;
  final VoidCallback onExport;

  const _ReactiveSelectionAppBar({
    required this.controller,
    required this.isCompact,
    required this.onExport,
  });

  @override
  Size get preferredSize => Size.fromHeight(
        kToolbarHeight + (isCompact ? 52.0 : 0.0),
      );

  @override
  Widget build(BuildContext context) {
    return SignalBuilder(
      builder: (context) {
        return ChatSelectionAppBar(
          selectedCount: controller.selectedMessageCount,
          selectableCount: controller.selectableMessageIndices.length,
          allSelected: controller.areAllMessagesSelected,
          isExporting: controller.isExporting.value,
          isCompact: isCompact,
          onClose: controller.exitSelectionMode,
          onToggleSelectAll: controller.toggleSelectAllMessages,
          onExport: onExport,
        );
      },
    );
  }
}

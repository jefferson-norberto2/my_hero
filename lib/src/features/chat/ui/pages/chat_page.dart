import 'package:flutter/material.dart';
import 'package:get_it/get_it.dart';
import 'package:go_router/go_router.dart';
import 'package:my_hero/src/core/routes/app_routes.dart';
import 'package:my_hero/src/features/chat/controller/chat_controller.dart';
import 'package:my_hero/src/features/chat/ui/widgets/chat_drawer.dart';
import 'package:my_hero/src/features/chat/ui/widgets/chat_input_bar.dart';
import 'package:my_hero/src/features/chat/ui/widgets/chat_message_list.dart';
import 'package:my_hero/src/features/settings/data/models/settings_args.dart';
import 'package:signals_flutter/signals_flutter.dart';

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

  void _showMetricsDetailsDialog(BuildContext context) {
    final metrics = _controller.currentMetrics.value;
    final theme = Theme.of(context);

    showDialog(
      context: context,
      builder: (ctx) {
        final usagePercent = metrics.contextUsagePercent;
        Color progressColor = Colors.teal;
        if (usagePercent > 80) {
          progressColor = Colors.red;
        } else if (usagePercent > 50) {
          progressColor = Colors.amber.shade800;
        }

        return AlertDialog(
          title: const Row(
            children: [
              Icon(Icons.analytics_outlined, color: Colors.teal),
              SizedBox(width: 8),
              Text('Métricas do Modelo'),
            ],
          ),
          content: Column(
            mainAxisSize: MainAxisSize.min,
            crossAxisAlignment: CrossAxisAlignment.stretch,
            children: [
              // VRAM Card
              Container(
                padding: const EdgeInsets.all(12),
                decoration: BoxDecoration(
                  color: Colors.purple.shade50,
                  borderRadius: BorderRadius.circular(10),
                  border: Border.all(color: Colors.purple.shade200),
                ),
                child: Row(
                  children: [
                    Icon(
                      Icons.memory_rounded,
                      color: Colors.purple.shade700,
                      size: 28,
                    ),
                    const SizedBox(width: 12),
                    Expanded(
                      child: Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          const Text(
                            'Memória de Vídeo (VRAM)',
                            style: TextStyle(
                              fontWeight: FontWeight.bold,
                              fontSize: 13,
                            ),
                          ),
                          Text(
                            'Alocado no Ollama: ${metrics.vramFormatted}',
                            style: TextStyle(
                              fontSize: 12,
                              color: Colors.purple.shade900,
                            ),
                          ),
                        ],
                      ),
                    ),
                  ],
                ),
              ),
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
                      '• Tokens utilizados: ${metrics.totalTokens} / ${metrics.contextLimit}',
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

  Widget _buildMetricsBadge(BuildContext context) {
    return SignalBuilder(
      builder: (context) {
        final isLoaded = _controller.isModelLoaded.value;
        if (!isLoaded) return const SizedBox.shrink();

        final metrics = _controller.currentMetrics.value;
        final usagePercent = metrics.contextUsagePercent;
        final remainingPercent = metrics.remainingPercent;

        Color badgeColor = Colors.teal;
        if (usagePercent > 80) {
          badgeColor = Colors.red.shade400;
        } else if (usagePercent > 50) {
          badgeColor = Colors.amber.shade700;
        }

        return Padding(
          padding: const EdgeInsets.symmetric(vertical: 8.0, horizontal: 4.0),
          child: InkWell(
            borderRadius: BorderRadius.circular(20),
            onTap: () => _showMetricsDetailsDialog(context),
            child: Container(
              padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 4),
              decoration: BoxDecoration(
                color: Theme.of(context)
                    .colorScheme
                    .surfaceContainerHighest
                    .withValues(alpha: 0.8),
                borderRadius: BorderRadius.circular(20),
                border: Border.all(color: badgeColor.withValues(alpha: 0.5)),
              ),
              child: Row(
                mainAxisSize: MainAxisSize.min,
                children: [
                  Icon(
                    Icons.memory_rounded,
                    size: 14,
                    color: Colors.purple.shade400,
                  ),
                  const SizedBox(width: 4),
                  Text(
                    metrics.vramFormatted,
                    style: const TextStyle(
                      fontSize: 11,
                      fontWeight: FontWeight.bold,
                    ),
                  ),
                  const SizedBox(width: 8),
                  Container(
                    width: 1,
                    height: 12,
                    color: Colors.grey.withValues(alpha: 0.4),
                  ),
                  const SizedBox(width: 8),
                  Icon(
                    Icons.pie_chart_outline_rounded,
                    size: 14,
                    color: badgeColor,
                  ),
                  const SizedBox(width: 4),
                  Text(
                    'Ctx: ${usagePercent.toStringAsFixed(0)}% (restam ${remainingPercent.toStringAsFixed(0)}%)',
                    style: TextStyle(
                      fontSize: 11,
                      fontWeight: FontWeight.bold,
                      color: badgeColor,
                    ),
                  ),
                ],
              ),
            ),
          ),
        );
      },
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      drawer: ChatDrawer(
        controller: _controller,
        onOpenSettings: () => _openSettings(context),
      ),
      appBar: AppBar(
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
        backgroundColor: Theme.of(context).colorScheme.inversePrimary,
        actions: [
          _buildMetricsBadge(context),
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
      ),
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
                    _controller.isGenerating.value) {
                  return const Padding(
                    padding: EdgeInsets.symmetric(vertical: 8.0),
                    child: LinearProgressIndicator(),
                  );
                }
                return const SizedBox.shrink();
              },
            ),
            ChatInputBar(controller: _controller),
          ],
        ),
      ),
    );
  }
}

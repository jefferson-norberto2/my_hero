import 'package:flutter/foundation.dart'
    show TargetPlatform, defaultTargetPlatform;
import 'package:flutter/material.dart';
import 'package:flutter_edge_ai/flutter_edge_ai.dart';
import 'package:signals_flutter/signals_flutter.dart';
import '../../../../core/constants/app_constants.dart';
import '../../controller/chat_controller.dart';

class ModelSelectorWidget extends StatelessWidget {
  final ChatController controller;

  const ModelSelectorWidget({super.key, required this.controller});

  IconData _backendIcon(PreferredBackend backend) {
    return switch (backend) {
      PreferredBackend.cpu => Icons.memory,
      PreferredBackend.gpu => Icons.speed,
      PreferredBackend.npu => Icons.auto_awesome,
    };
  }

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final isMobile = defaultTargetPlatform == TargetPlatform.android ||
        defaultTargetPlatform == TargetPlatform.iOS;

    return SignalBuilder(
      builder: (context) {
        final isLoading = controller.isLoading.value;
        final selectedBackend = controller.selectedBackend.value;
        final availableBackends = controller.availableBackends;
        final currentMaxTokens = controller.maxTokens.value;
        final enableThinking = controller.enableThinking.value;
        final progress = controller.downloadProgress.value;
        final status = controller.loadingStatus.value;

        return Column(
          crossAxisAlignment: CrossAxisAlignment.center,
          mainAxisSize: MainAxisSize.min,
          children: [
            const Text(
              'Configure e carregue o modelo Gemma 4:',
              style: TextStyle(fontWeight: FontWeight.bold),
            ),
            const SizedBox(height: 10),
            SegmentedButton<PreferredBackend>(
              segments: availableBackends
                  .map(
                    (backend) => ButtonSegment<PreferredBackend>(
                      value: backend,
                      label: Text(AppConstants.backendLabel(backend)),
                      icon: Icon(_backendIcon(backend), size: 18),
                    ),
                  )
                  .toList(),
              selected: {selectedBackend},
              onSelectionChanged: isLoading
                  ? null
                  : (selection) => controller.selectBackend(selection.first),
            ),
            if (selectedBackend == PreferredBackend.npu) ...[
              const SizedBox(height: 6),
              Text(
                'NPU requer hardware compatível (Snapdragon / Google Tensor / Intel NPU) e variante NPU; em bundles genéricos, o runtime faz fallback automático para GPU/CPU.',
                textAlign: TextAlign.center,
                style: theme.textTheme.bodySmall?.copyWith(
                  color: theme.colorScheme.onSurfaceVariant,
                ),
              ),
            ],
            const SizedBox(height: 14),
            Text(
              'Janela de contexto: ${AppConstants.formatTokenCount(currentMaxTokens)} tokens (máx. do modelo: ${AppConstants.formatTokenCount(controller.maxSupportedContextTokens)})',
              style: theme.textTheme.labelMedium?.copyWith(
                fontWeight: FontWeight.w600,
              ),
            ),
            const SizedBox(height: 6),
            Wrap(
              spacing: 6,
              runSpacing: 4,
              alignment: WrapAlignment.center,
              children: AppConstants.contextTokenPresets.map((tokens) {
                final isSelected = currentMaxTokens == tokens;
                return ChoiceChip(
                  label: Text(AppConstants.formatTokenCount(tokens)),
                  selected: isSelected,
                  onSelected: isLoading
                      ? null
                      : (_) => controller.setMaxTokens(tokens),
                );
              }).toList(),
            ),
            if (currentMaxTokens < AppConstants.recommendedMinVisionTokens) ...[
              const SizedBox(height: 4),
              Text(
                'Para envio de imagens, recomenda-se ao menos 4K tokens.',
                style: theme.textTheme.bodySmall?.copyWith(
                  color: theme.colorScheme.onSurfaceVariant,
                ),
              ),
            ] else if (isMobile && currentMaxTokens >= 16384) ...[
              const SizedBox(height: 4),
              Text(
                'Atenção: contextos de 16K–32K consomem bastante RAM no mobile.',
                style: theme.textTheme.bodySmall?.copyWith(
                  color: theme.colorScheme.error,
                ),
              ),
            ],
            const SizedBox(height: 8),
            ConstrainedBox(
              constraints: const BoxConstraints(maxWidth: 420),
              child: SwitchListTile.adaptive(
                dense: true,
                contentPadding: const EdgeInsets.symmetric(horizontal: 8),
                secondary: const Icon(
                  Icons.psychology_alt_outlined,
                  color: Colors.teal,
                ),
                title: const Text('Modo Pensamento (Thinking)'),
                subtitle: const Text(
                  'Exibe o raciocínio do Gemma 4 antes da resposta final',
                  style: TextStyle(fontSize: 12),
                ),
                value: enableThinking,
                onChanged: isLoading ? null : controller.setEnableThinking,
              ),
            ),
            const SizedBox(height: 12),
            Row(
              mainAxisAlignment: MainAxisAlignment.center,
              children: [
                ElevatedButton(
                  onPressed: isLoading
                      ? null
                      : () => controller.downloadAndLoadModel(is2B: true),
                  child: const Text('Carregar 2B'),
                ),
                const SizedBox(width: 16),
                ElevatedButton(
                  onPressed: isLoading
                      ? null
                      : () => controller.downloadAndLoadModel(is2B: false),
                  child: const Text('Carregar 4B'),
                ),
              ],
            ),
            if (isLoading) ...[
              const SizedBox(height: 18),
              Text(
                status,
                textAlign: TextAlign.center,
                style: const TextStyle(fontWeight: FontWeight.w500),
              ),
              const SizedBox(height: 8),
              LinearProgressIndicator(value: progress > 0 ? progress : null),
              const SizedBox(height: 6),
              Row(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  Text('${(progress * 100).toStringAsFixed(1)}%'),
                  if (progress < 1.0) ...[
                    const SizedBox(width: 12),
                    TextButton.icon(
                      onPressed: controller.cancelDownload,
                      icon: const Icon(Icons.close, size: 16),
                      label: const Text('Cancelar'),
                    ),
                  ],
                ],
              ),
            ],
            const Divider(height: 28),
          ],
        );
      },
    );
  }
}

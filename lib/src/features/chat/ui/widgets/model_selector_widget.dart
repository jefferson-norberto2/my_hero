import 'package:flutter/foundation.dart'
    show TargetPlatform, defaultTargetPlatform;
import 'package:material_ui/material_ui.dart';
import 'package:flutter_edge_ai/flutter_edge_ai.dart';
import 'package:signals_flutter/signals_flutter.dart';
import 'package:my_hero/src/core/constants/app_constants.dart';
import 'package:my_hero/src/features/chat/controller/chat_controller.dart';
import 'package:my_hero/src/features/chat/ui/widgets/model_catalog_list.dart';

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
        final selectedModelId = controller.selectedModelId.value;
        final loadedModelId = controller.loadedModelId.value;
        final installedFiles = controller.installedModelFiles.value;
        final deletingFile = controller.deletingModelFileName.value;
        final selectedSpec = controller.currentModelSpec;
        final isSelectedInstalled = installedFiles.contains(selectedSpec.fileName);
        final selectedBackend = controller.selectedBackend.value;
        final availableBackends = controller.availableBackends;
        final currentMaxTokens = controller.maxTokens.value;
        final enableThinking = controller.enableThinking.value;
        final progress = controller.downloadProgress.value;
        final status = controller.loadingStatus.value;

        return ConstrainedBox(
          constraints: const BoxConstraints(maxWidth: 560),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.center,
            mainAxisSize: MainAxisSize.min,
            children: [
              const Text(
                'Escolha um modelo aberto do catálogo:',
                style: TextStyle(fontWeight: FontWeight.bold, fontSize: 15),
              ),
              const SizedBox(height: 8),
              ModelCatalogList(
                models: controller.availableModels,
                selectedModelId: selectedModelId,
                installedFileNames: installedFiles,
                loadedModelId: loadedModelId,
                deletingFileName: deletingFile,
                enabled: !isLoading,
                maxHeight: 240,
                onSelect: (spec) => controller.selectModel(spec.id),
                onDelete: controller.uninstallModel,
              ),
              const SizedBox(height: 14),
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
                'Janela de contexto: ${AppConstants.formatTokenCount(currentMaxTokens)} tokens (máx. do modelo: ${AppConstants.formatTokenCount(selectedSpec.maxContextTokens)})',
                style: theme.textTheme.labelMedium?.copyWith(
                  fontWeight: FontWeight.w600,
                ),
              ),
              const SizedBox(height: 6),
              Wrap(
                spacing: 6,
                runSpacing: 4,
                alignment: WrapAlignment.center,
                children: selectedSpec.contextTokenPresets.map((tokens) {
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
              if (selectedSpec.supportsImage &&
                  currentMaxTokens < AppConstants.recommendedMinVisionTokens) ...[
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
              if (selectedSpec.supportsThinking)
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
                    subtitle: Text(
                      'Exibe o raciocínio de ${selectedSpec.displayName} antes da resposta final',
                      style: const TextStyle(fontSize: 12),
                    ),
                    value: enableThinking,
                    onChanged: isLoading ? null : controller.setEnableThinking,
                  ),
                )
              else
                Padding(
                  padding: const EdgeInsets.symmetric(vertical: 4),
                  child: Text(
                    '${selectedSpec.displayName} não possui Modo Pensamento (Thinking).',
                    style: theme.textTheme.bodySmall?.copyWith(
                      color: theme.colorScheme.onSurfaceVariant,
                    ),
                  ),
                ),
              const SizedBox(height: 12),
              FilledButton.icon(
                onPressed: isLoading
                    ? null
                    : () => controller.downloadAndLoadModel(
                          modelId: selectedSpec.id,
                        ),
                icon: Icon(
                  isSelectedInstalled
                      ? Icons.play_arrow_rounded
                      : Icons.download_rounded,
                ),
                label: Text(
                  isSelectedInstalled
                      ? 'Carregar ${selectedSpec.displayName}'
                      : 'Baixar e Carregar ${selectedSpec.displayName} (${selectedSpec.formattedSize})',
                ),
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
          ),
        );
      },
    );
  }
}

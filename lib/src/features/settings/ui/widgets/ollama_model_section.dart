import 'package:flutter/material.dart';
import 'package:my_hero/src/core/theme/app_theme.dart';
import 'package:my_hero/src/shared/models/model_download_state.dart';
import 'package:my_hero/src/shared/models/ollama_model.dart';

/// Card section displaying the Ollama model dropdown, refresh button,
/// model download (`/api/pull`) controls, and memory load control.
class OllamaModelSection extends StatelessWidget {
  final bool isConnected;
  final String serverUrl;
  final List<OllamaModel> models;
  final String selectedModelName;
  final bool isRefreshing;
  final bool isLoadingModel;
  final double loadingProgress;
  final String loadingStatus;
  final String? loadErrorMessage;
  final bool isModelLoaded;
  final ModelDownloadState downloadState;
  final TextEditingController? pullModelController;
  final VoidCallback onRefreshModels;
  final ValueChanged<String> onModelSelected;
  final VoidCallback onLoadModel;
  final ValueChanged<String>? onDownloadModel;
  final VoidCallback? onCancelDownload;

  const OllamaModelSection({
    super.key,
    required this.isConnected,
    required this.serverUrl,
    required this.models,
    required this.selectedModelName,
    required this.isRefreshing,
    required this.isLoadingModel,
    required this.loadingProgress,
    required this.loadingStatus,
    required this.loadErrorMessage,
    required this.isModelLoaded,
    this.downloadState = const ModelDownloadState.idle(),
    this.pullModelController,
    required this.onRefreshModels,
    required this.onModelSelected,
    required this.onLoadModel,
    this.onDownloadModel,
    this.onCancelDownload,
  });

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final appColors = AppThemeColors.of(context);

    return Card(
      elevation: 0,
      color: appColors.sectionCardBackground,
      shape: RoundedRectangleBorder(
        borderRadius: BorderRadius.circular(16),
        side: BorderSide(color: appColors.sectionCardBorder),
      ),
      child: Padding(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            Row(
              children: [
                Icon(Icons.hub_outlined, color: appColors.accent),
                const SizedBox(width: 8),
                const Expanded(
                  child: Text(
                    'Modelo Ollama',
                    style: TextStyle(fontWeight: FontWeight.bold, fontSize: 15),
                  ),
                ),
                if (isRefreshing)
                  const Padding(
                    padding: EdgeInsets.all(10.0),
                    child: SizedBox(
                      width: 18,
                      height: 18,
                      child: CircularProgressIndicator(strokeWidth: 2),
                    ),
                  )
                else
                  IconButton(
                    icon: const Icon(Icons.refresh, size: 20),
                    tooltip: 'Atualizar lista de modelos',
                    onPressed: isLoadingModel ? null : onRefreshModels,
                  ),
              ],
            ),
            const SizedBox(height: 12),
            if (!isConnected)
              _buildDisconnectedState(theme)
            else ...[
              if (models.isEmpty)
                _buildEmptyState(theme, appColors)
              else
                _buildModelControls(theme, appColors),
              const Divider(height: 24),
              _buildModelPullSection(theme, appColors),
            ],
            if (loadErrorMessage != null) ...[
              const SizedBox(height: 12),
              _buildErrorBanner(theme, loadErrorMessage!),
            ],
          ],
        ),
      ),
    );
  }

  Widget _buildDisconnectedState(ThemeData theme) {
    return Container(
      padding: const EdgeInsets.all(12),
      decoration: BoxDecoration(
        color: theme.colorScheme.errorContainer.withValues(alpha: 0.4),
        borderRadius: BorderRadius.circular(12),
        border: Border.all(
          color: theme.colorScheme.error.withValues(alpha: 0.3),
        ),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Icon(Icons.cloud_off, color: theme.colorScheme.error, size: 24),
              const SizedBox(width: 10),
              Expanded(
                child: Text(
                  'Ollama não detectado',
                  style: theme.textTheme.titleSmall?.copyWith(
                    fontWeight: FontWeight.bold,
                    color: theme.colorScheme.error,
                  ),
                ),
              ),
            ],
          ),
          const SizedBox(height: 8),
          Text(
            'Não foi possível conectar em $serverUrl. Verifique se o serviço está ativo ou altere o IP abaixo.',
            style: const TextStyle(fontSize: 13),
          ),
          const SizedBox(height: 12),
          FilledButton.icon(
            onPressed: isRefreshing ? null : onRefreshModels,
            icon: const Icon(Icons.refresh),
            label: const Text('Tentar Novamente'),
          ),
        ],
      ),
    );
  }

  Widget _buildEmptyState(ThemeData theme, AppThemeColors appColors) {
    return Column(
      children: [
        Icon(Icons.psychology_alt, color: appColors.accent, size: 36),
        const SizedBox(height: 8),
        const Text(
          'Nenhum modelo encontrado no Ollama',
          style: TextStyle(fontWeight: FontWeight.bold, fontSize: 15),
          textAlign: TextAlign.center,
        ),
        const SizedBox(height: 6),
        Text(
          'Baixe um modelo diretamente abaixo ou pelo terminal (ex: llama3.2:3b).',
          textAlign: TextAlign.center,
          style: TextStyle(
            fontSize: 12,
            color: theme.colorScheme.onSurfaceVariant,
          ),
        ),
        const SizedBox(height: 12),
        FilledButton.icon(
          onPressed: isRefreshing ? null : onRefreshModels,
          icon: const Icon(Icons.refresh),
          label: const Text('Atualizar Lista'),
        ),
      ],
    );
  }

  Widget _buildModelControls(ThemeData theme, AppThemeColors appColors) {
    final effectiveSelected = models.any((m) => m.name == selectedModelName)
        ? selectedModelName
        : models.first.name;
    final selectedModelObj =
        models.where((m) => m.name == effectiveSelected).firstOrNull ??
            models.first;
    final isSelectedLoaded = selectedModelObj.isLoaded ||
        (selectedModelObj.name == selectedModelName && isModelLoaded);
    final inputFillColor = theme.brightness == Brightness.dark
        ? theme.colorScheme.surfaceContainerHighest
        : theme.colorScheme.surface;

    return Column(
      crossAxisAlignment: CrossAxisAlignment.stretch,
      children: [
        DropdownButtonFormField<String>(
          key: ValueKey(
            '${effectiveSelected}_${models.map((m) => '${m.name}:${m.isLoaded}').join(',')}',
          ),
          isExpanded: true,
          initialValue: effectiveSelected,
          dropdownColor: theme.colorScheme.surface,
          decoration: InputDecoration(
            contentPadding:
                const EdgeInsets.symmetric(horizontal: 14, vertical: 12),
            border: OutlineInputBorder(borderRadius: BorderRadius.circular(12)),
            labelText: 'Selecione o modelo',
            filled: true,
            fillColor: inputFillColor,
          ),
          selectedItemBuilder: (context) {
            return models.map((OllamaModel model) {
              final loaded = model.isLoaded ||
                  (model.name == selectedModelName && isModelLoaded);
              return Align(
                alignment: Alignment.centerLeft,
                child: Text(
                  '${model.name}${model.parameterSize != null ? " (${model.parameterSize})" : ""}${loaded ? " • Carregado" : ""}',
                  overflow: TextOverflow.ellipsis,
                  style: const TextStyle(fontWeight: FontWeight.w600),
                ),
              );
            }).toList();
          },
          items: models.map((OllamaModel model) {
            final loaded = model.isLoaded ||
                (model.name == selectedModelName && isModelLoaded);
            return DropdownMenuItem<String>(
              value: model.name,
              child: Row(
                children: [
                  Flexible(
                    child: Text(
                      model.name,
                      overflow: TextOverflow.ellipsis,
                      style: const TextStyle(fontWeight: FontWeight.w600),
                    ),
                  ),
                  if (model.parameterSize != null) ...[
                    const SizedBox(width: 8),
                    Container(
                      padding: const EdgeInsets.symmetric(
                        horizontal: 6,
                        vertical: 2,
                      ),
                      decoration: BoxDecoration(
                        color: appColors.accent.withValues(alpha: 0.15),
                        borderRadius: BorderRadius.circular(6),
                      ),
                      child: Text(
                        model.parameterSize!,
                        style: TextStyle(fontSize: 11, color: appColors.accent),
                      ),
                    ),
                  ],
                  if (loaded) ...[
                    const SizedBox(width: 6),
                    Container(
                      padding: const EdgeInsets.symmetric(
                        horizontal: 6,
                        vertical: 2,
                      ),
                      decoration: BoxDecoration(
                        color: appColors.success.withValues(alpha: 0.18),
                        borderRadius: BorderRadius.circular(6),
                      ),
                      child: Row(
                        mainAxisSize: MainAxisSize.min,
                        children: [
                          Icon(Icons.bolt, size: 12, color: appColors.success),
                          const SizedBox(width: 2),
                          Text(
                            'Carregado',
                            style: TextStyle(
                              fontSize: 10,
                              color: appColors.success,
                              fontWeight: FontWeight.bold,
                            ),
                          ),
                        ],
                      ),
                    ),
                  ],
                ],
              ),
            );
          }).toList(),
          onChanged: isLoadingModel
              ? null
              : (newVal) {
                  if (newVal != null) {
                    onModelSelected(newVal);
                  }
                },
        ),
        const SizedBox(height: 12),
        Row(
          children: [
            if (isSelectedLoaded)
              Expanded(
                child: Row(
                  children: [
                    Icon(Icons.check_circle, color: appColors.success, size: 18),
                    const SizedBox(width: 6),
                    Expanded(
                      child: Text(
                        'Já carregado na memória do PC',
                        style: TextStyle(
                          color: appColors.success,
                          fontWeight: FontWeight.bold,
                          fontSize: 12,
                        ),
                      ),
                    ),
                  ],
                ),
              )
            else
              Expanded(
                child: Text(
                  'Será carregado na memória ao usar',
                  style: TextStyle(
                    fontSize: 12,
                    color: theme.colorScheme.onSurfaceVariant,
                  ),
                ),
              ),
            const SizedBox(width: 12),
            FilledButton.icon(
              onPressed: isLoadingModel ? null : onLoadModel,
              icon: Icon(isSelectedLoaded ? Icons.play_arrow : Icons.memory),
              label: Text(
                isSelectedLoaded ? 'Usar Modelo' : 'Carregar na Memória',
              ),
            ),
          ],
        ),
        if (isLoadingModel) ...[
          const SizedBox(height: 12),
          LinearProgressIndicator(
            value: loadingProgress > 0 ? loadingProgress : null,
          ),
          const SizedBox(height: 6),
          Text(
            loadingStatus,
            style: const TextStyle(fontSize: 12, fontStyle: FontStyle.italic),
          ),
        ],
      ],
    );
  }

  Widget _buildModelPullSection(ThemeData theme, AppThemeColors appColors) {
    final isDownloading = downloadState.isActive;
    final percent = downloadState.progressPercent;
    final inputFillColor = theme.brightness == Brightness.dark
        ? theme.colorScheme.surfaceContainerHighest
        : theme.colorScheme.surface;

    return Column(
      crossAxisAlignment: CrossAxisAlignment.stretch,
      children: [
        if (!isDownloading) ...[
          Row(
            children: [
              Expanded(
                child: TextField(
                  key: const Key('pull_model_name_field'),
                  controller: pullModelController,
                  enabled: onDownloadModel != null,
                  textInputAction: TextInputAction.done,
                  decoration: InputDecoration(
                    isDense: true,
                    contentPadding: const EdgeInsets.symmetric(
                      horizontal: 12,
                      vertical: 11,
                    ),
                    border: OutlineInputBorder(
                      borderRadius: BorderRadius.circular(12),
                    ),
                    labelText: 'Baixar novo modelo',
                    hintText: 'Ex: llama3.2:3b',
                    filled: true,
                    fillColor: inputFillColor,
                  ),
                  onSubmitted: (value) {
                    if (value.trim().isNotEmpty) {
                      onDownloadModel?.call(value.trim());
                    }
                  },
                ),
              ),
              const SizedBox(width: 8),
              FilledButton.tonalIcon(
                key: const Key('pull_model_button'),
                onPressed: onDownloadModel == null
                    ? null
                    : () {
                        final target = pullModelController?.text.trim() ?? '';
                        if (target.isNotEmpty) {
                          onDownloadModel!(target);
                        }
                      },
                icon: const Icon(Icons.download, size: 18),
                label: const Text('Baixar'),
              ),
            ],
          ),
          if (downloadState.status == ModelDownloadStatus.completed &&
              downloadState.statusMessage.isNotEmpty) ...[
            const SizedBox(height: 8),
            Text(
              downloadState.statusMessage,
              style: TextStyle(
                fontSize: 12,
                color: appColors.success,
                fontWeight: FontWeight.w600,
              ),
            ),
          ] else if (downloadState.status == ModelDownloadStatus.cancelled &&
              downloadState.statusMessage.isNotEmpty) ...[
            const SizedBox(height: 8),
            Text(
              downloadState.statusMessage,
              style: TextStyle(
                fontSize: 12,
                color: theme.colorScheme.onSurfaceVariant,
              ),
            ),
          ],
        ] else ...[
          Row(
            children: [
              Icon(Icons.downloading, color: appColors.accent, size: 18),
              const SizedBox(width: 8),
              Expanded(
                child: Text(
                  percent != null
                      ? 'Baixando ${downloadState.modelName}: $percent%'
                      : 'Baixando ${downloadState.modelName}...',
                  style: const TextStyle(
                    fontWeight: FontWeight.w600,
                    fontSize: 13,
                  ),
                ),
              ),
              TextButton.icon(
                key: const Key('cancel_model_download_button'),
                onPressed: onCancelDownload,
                icon: const Icon(Icons.close, size: 16),
                label: const Text('Cancelar'),
              ),
            ],
          ),
          const SizedBox(height: 6),
          LinearProgressIndicator(
            value: downloadState.totalBytes > 0 ? downloadState.progress : null,
          ),
          if (downloadState.statusMessage.isNotEmpty) ...[
            const SizedBox(height: 6),
            Text(
              downloadState.statusMessage,
              style: const TextStyle(fontSize: 12, fontStyle: FontStyle.italic),
            ),
          ],
        ],
      ],
    );
  }

  Widget _buildErrorBanner(ThemeData theme, String message) {
    return Container(
      padding: const EdgeInsets.all(10),
      decoration: BoxDecoration(
        color: theme.colorScheme.errorContainer.withValues(alpha: 0.5),
        borderRadius: BorderRadius.circular(10),
        border: Border.all(
          color: theme.colorScheme.error.withValues(alpha: 0.4),
        ),
      ),
      child: Row(
        children: [
          Icon(Icons.error_outline, color: theme.colorScheme.error, size: 18),
          const SizedBox(width: 8),
          Expanded(
            child: Text(
              message,
              style: TextStyle(
                fontSize: 12,
                color: theme.colorScheme.onErrorContainer,
              ),
            ),
          ),
        ],
      ),
    );
  }
}

import 'package:flutter/material.dart';
import 'package:my_hero/src/shared/models/ollama_model.dart';

/// Card section displaying the Ollama model dropdown, refresh button, and memory load control.
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
  final VoidCallback onRefreshModels;
  final ValueChanged<String> onModelSelected;
  final VoidCallback onLoadModel;

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
    required this.onRefreshModels,
    required this.onModelSelected,
    required this.onLoadModel,
  });

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);

    return Card(
      elevation: 0,
      color: theme.colorScheme.surfaceContainerHighest.withValues(alpha: 0.4),
      shape: RoundedRectangleBorder(
        borderRadius: BorderRadius.circular(16),
        side: BorderSide(
          color: theme.colorScheme.outlineVariant.withValues(alpha: 0.5),
        ),
      ),
      child: Padding(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            Row(
              children: [
                const Icon(Icons.hub_outlined, color: Colors.teal),
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
            else if (models.isEmpty)
              _buildEmptyState()
            else
              _buildModelControls(theme),
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

  Widget _buildEmptyState() {
    return Column(
      children: [
        const Icon(Icons.psychology_alt, color: Colors.teal, size: 36),
        const SizedBox(height: 8),
        const Text(
          'Nenhum modelo encontrado no Ollama',
          style: TextStyle(fontWeight: FontWeight.bold, fontSize: 15),
          textAlign: TextAlign.center,
        ),
        const SizedBox(height: 6),
        const Text(
          'Você pode baixar um modelo usando seu terminal:\nEx: ollama pull llama3.2 ou ollama pull deepseek-r1:8b',
          textAlign: TextAlign.center,
          style: TextStyle(fontSize: 12, color: Colors.grey),
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

  Widget _buildModelControls(ThemeData theme) {
    final effectiveSelected = models.any((m) => m.name == selectedModelName)
        ? selectedModelName
        : models.first.name;
    final selectedModelObj =
        models.where((m) => m.name == effectiveSelected).firstOrNull ??
            models.first;
    final isSelectedLoaded = selectedModelObj.isLoaded ||
        (selectedModelObj.name == selectedModelName && isModelLoaded);

    return Column(
      crossAxisAlignment: CrossAxisAlignment.stretch,
      children: [
        DropdownButtonFormField<String>(
          key: ValueKey(
            '${effectiveSelected}_${models.map((m) => '${m.name}:${m.isLoaded}').join(',')}',
          ),
          isExpanded: true,
          initialValue: effectiveSelected,
          decoration: InputDecoration(
            contentPadding:
                const EdgeInsets.symmetric(horizontal: 14, vertical: 12),
            border: OutlineInputBorder(borderRadius: BorderRadius.circular(12)),
            labelText: 'Selecione o modelo',
            filled: true,
            fillColor: theme.colorScheme.surface,
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
                        color: Colors.teal.withValues(alpha: 0.15),
                        borderRadius: BorderRadius.circular(6),
                      ),
                      child: Text(
                        model.parameterSize!,
                        style: const TextStyle(fontSize: 11, color: Colors.teal),
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
                        color: Colors.green.withValues(alpha: 0.2),
                        borderRadius: BorderRadius.circular(6),
                      ),
                      child: const Row(
                        mainAxisSize: MainAxisSize.min,
                        children: [
                          Icon(Icons.bolt, size: 12, color: Colors.green),
                          SizedBox(width: 2),
                          Text(
                            'Carregado',
                            style: TextStyle(
                              fontSize: 10,
                              color: Colors.green,
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
              const Expanded(
                child: Row(
                  children: [
                    Icon(Icons.check_circle, color: Colors.green, size: 18),
                    SizedBox(width: 6),
                    Expanded(
                      child: Text(
                        'Já carregado na memória do PC',
                        style: TextStyle(
                          color: Colors.green,
                          fontWeight: FontWeight.bold,
                          fontSize: 12,
                        ),
                      ),
                    ),
                  ],
                ),
              )
            else
              const Expanded(
                child: Text(
                  'Será carregado na memória ao usar',
                  style: TextStyle(fontSize: 12, color: Colors.grey),
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

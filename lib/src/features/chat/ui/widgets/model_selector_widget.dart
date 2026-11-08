import 'package:flutter/material.dart';
import 'package:signals_flutter/signals_flutter.dart';
import '../../controller/chat_controller.dart';
import '../../data/adapter/ollama_model.dart';

class ModelSelectorWidget extends StatelessWidget {
  final ChatController controller;

  const ModelSelectorWidget({super.key, required this.controller});

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);

    return SignalBuilder(
      builder: (context) {
        final isConnected = controller.isOllamaConnected.value;
        final models = controller.availableModels.value;
        final selected = controller.selectedModelName.value;
        final isLoading = controller.isLoading.value;
        final status = controller.loadingStatus.value;
        final progress = controller.loadingProgress.value;

        if (!isConnected) {
          return Card(
            color: theme.colorScheme.errorContainer.withValues(alpha: 0.4),
            shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(16)),
            child: Padding(
              padding: const EdgeInsets.all(16.0),
              child: Column(
                children: [
                  Row(
                    children: [
                      Icon(Icons.cloud_off, color: theme.colorScheme.error, size: 28),
                      const SizedBox(width: 12),
                      Expanded(
                        child: Text(
                          'Ollama não detectado',
                          style: theme.textTheme.titleMedium?.copyWith(
                            fontWeight: FontWeight.bold,
                            color: theme.colorScheme.error,
                          ),
                        ),
                      ),
                    ],
                  ),
                  const SizedBox(height: 8),
                  const Text(
                    'Verifique se o Ollama está em execução no seu computador (porta 11434). Você pode iniciá-lo no terminal com `ollama serve`.',
                    style: TextStyle(fontSize: 13),
                  ),
                  const SizedBox(height: 12),
                  FilledButton.icon(
                    onPressed: controller.refreshModels,
                    icon: const Icon(Icons.refresh),
                    label: const Text('Tentar Conectar Novamente'),
                  ),
                ],
              ),
            ),
          );
        }

        if (models.isEmpty) {
          return Card(
            shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(16)),
            child: Padding(
              padding: const EdgeInsets.all(16.0),
              child: Column(
                children: [
                  const Icon(Icons.psychology_alt, color: Colors.teal, size: 36),
                  const SizedBox(height: 8),
                  const Text(
                    'Nenhum modelo encontrado no Ollama',
                    style: TextStyle(fontWeight: FontWeight.bold, fontSize: 16),
                  ),
                  const SizedBox(height: 6),
                  const Text(
                    'Você pode baixar um modelo usando seu terminal:\nEx: `ollama pull llama3.2` ou `ollama pull deepseek-r1:8b`',
                    textAlign: TextAlign.center,
                    style: TextStyle(fontSize: 12, color: Colors.grey),
                  ),
                  const SizedBox(height: 12),
                  FilledButton.icon(
                    onPressed: controller.refreshModels,
                    icon: const Icon(Icons.refresh),
                    label: const Text('Atualizar Lista de Modelos'),
                  ),
                ],
              ),
            ),
          );
        }

        final selectedModelObj = models.where((m) => m.name == selected).firstOrNull ?? models.first;
        final isSelectedLoaded = selectedModelObj.isLoaded;

        return Card(
          elevation: 0,
          color: theme.colorScheme.surfaceContainerHighest.withValues(alpha: 0.5),
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
                        'Modelos disponíveis no seu Ollama:',
                        style: TextStyle(fontWeight: FontWeight.bold, fontSize: 15),
                      ),
                    ),
                    IconButton(
                      icon: const Icon(Icons.refresh, size: 20),
                      tooltip: 'Atualizar modelos',
                      onPressed: isLoading ? null : controller.refreshModels,
                    ),
                  ],
                ),
                const SizedBox(height: 12),
                DropdownButtonFormField<String>(
                  isExpanded: true,
                  initialValue: models.any((m) => m.name == selected) ? selected : models.first.name,
                  decoration: InputDecoration(
                    contentPadding: const EdgeInsets.symmetric(horizontal: 14, vertical: 12),
                    border: OutlineInputBorder(borderRadius: BorderRadius.circular(12)),
                    labelText: 'Selecione o modelo',
                    filled: true,
                    fillColor: theme.colorScheme.surface,
                  ),
                  selectedItemBuilder: (context) {
                    return models.map((OllamaModel model) {
                      return Align(
                        alignment: Alignment.centerLeft,
                        child: Text(
                          '${model.name}${model.parameterSize != null ? " (${model.parameterSize})" : ""}${model.isLoaded ? " • Carregado" : ""}',
                          overflow: TextOverflow.ellipsis,
                          style: const TextStyle(fontWeight: FontWeight.w600),
                        ),
                      );
                    }).toList();
                  },
                  items: models.map((OllamaModel model) {
                    return DropdownMenuItem<String>(
                      value: model.name,
                      child: Row(
                        mainAxisSize: MainAxisSize.min,
                        children: [
                          Text(
                            model.name,
                            style: const TextStyle(fontWeight: FontWeight.w600),
                          ),
                          if (model.parameterSize != null) ...[
                            const SizedBox(width: 8),
                            Container(
                              padding: const EdgeInsets.symmetric(horizontal: 6, vertical: 2),
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
                          if (model.isLoaded) ...[
                            const SizedBox(width: 6),
                            Container(
                              padding: const EdgeInsets.symmetric(horizontal: 6, vertical: 2),
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
                  onChanged: isLoading
                      ? null
                      : (newVal) {
                          if (newVal != null) {
                            controller.selectedModelName.value = newVal;
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
                    FilledButton.icon(
                      onPressed: isLoading
                          ? null
                          : () {
                              final target = controller.selectedModelName.value.isNotEmpty
                                  ? controller.selectedModelName.value
                                  : models.first.name;
                              controller.loadModel(target);
                            },
                      icon: Icon(isSelectedLoaded ? Icons.play_arrow : Icons.memory),
                      label: Text(isSelectedLoaded ? 'Usar Modelo' : 'Carregar na Memória'),
                    ),
                  ],
                ),
                if (isLoading) ...[
                  const SizedBox(height: 12),
                  LinearProgressIndicator(value: progress > 0 ? progress : null),
                  const SizedBox(height: 6),
                  Text(
                    status,
                    style: const TextStyle(fontSize: 12, fontStyle: FontStyle.italic),
                  ),
                ],
              ],
            ),
          ),
        );
      },
    );
  }
}

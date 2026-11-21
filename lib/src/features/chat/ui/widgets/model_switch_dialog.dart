import 'package:flutter/foundation.dart'
    show TargetPlatform, defaultTargetPlatform;
import 'package:flutter/material.dart';
import 'package:flutter_edge_ai/flutter_edge_ai.dart';
import 'package:signals_flutter/signals_flutter.dart';
import '../../../../core/constants/app_constants.dart';
import '../../controller/chat_controller.dart';
import '../../data/adapter/response_style.dart';
import 'model_catalog_list.dart';

class ModelSwitchDialog extends StatefulWidget {
  final ChatController controller;

  const ModelSwitchDialog({super.key, required this.controller});

  static Future<void> show(BuildContext context, ChatController controller) {
    return showDialog(
      context: context,
      builder: (_) => ModelSwitchDialog(controller: controller),
    );
  }

  @override
  State<ModelSwitchDialog> createState() => _ModelSwitchDialogState();
}

class _ModelSwitchDialogState extends State<ModelSwitchDialog> {
  late String _selectedModelId;
  late PreferredBackend _selectedBackend;
  late int _selectedMaxTokens;
  late bool _selectedEnableThinking;
  late ResponseStyle _selectedResponseStyle;

  @override
  void initState() {
    super.initState();
    _selectedModelId = widget.controller.selectedModelId.value;
    _selectedBackend = widget.controller.selectedBackend.value;
    _selectedMaxTokens = widget.controller.maxTokens.value;
    _selectedEnableThinking = widget.controller.enableThinking.value;
    _selectedResponseStyle = widget.controller.responseStyle.value;
  }

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
    final availableBackends = widget.controller.availableBackends;
    final modelSpec = AppConstants.modelSpecById(_selectedModelId);
    final isMobile = defaultTargetPlatform == TargetPlatform.android ||
        defaultTargetPlatform == TargetPlatform.iOS;

    return AlertDialog(
      title: const Row(
        children: [
          Icon(Icons.tune, color: Colors.teal),
          SizedBox(width: 8),
          Expanded(
            child: Text('Configurações da Conversa'),
          ),
        ],
      ),
      content: SizedBox(
        width: 440,
        child: SingleChildScrollView(
          child: SignalBuilder(
            builder: (context) {
              final installedFiles =
                  widget.controller.installedModelFiles.value;
              final loadedModelId = widget.controller.loadedModelId.value;
              final deletingFile =
                  widget.controller.deletingModelFileName.value;

              return Column(
                mainAxisSize: MainAxisSize.min,
                crossAxisAlignment: CrossAxisAlignment.stretch,
                children: [
                  const Text(
                    'Modelo do catálogo:',
                    style: TextStyle(fontWeight: FontWeight.bold),
                  ),
                  const SizedBox(height: 8),
                  ModelCatalogList(
                    models: widget.controller.availableModels,
                    selectedModelId: _selectedModelId,
                    installedFileNames: installedFiles,
                    loadedModelId: loadedModelId,
                    deletingFileName: deletingFile,
                    maxHeight: 190,
                    onSelect: (spec) {
                      setState(() {
                        _selectedModelId = spec.id;
                        _selectedMaxTokens = AppConstants.clampMaxTokens(
                          _selectedMaxTokens,
                          modelId: spec.id,
                        );
                        if (!spec.supportsThinking) {
                          _selectedEnableThinking = false;
                        }
                      });
                    },
                    onDelete: widget.controller.uninstallModel,
                  ),
                  const SizedBox(height: 16),
                  const Text(
                    'Backend de execução:',
                    style: TextStyle(fontWeight: FontWeight.bold),
                  ),
                  const SizedBox(height: 8),
                  SegmentedButton<PreferredBackend>(
                    segments: availableBackends
                        .map(
                          (backend) => ButtonSegment<PreferredBackend>(
                            value: backend,
                            label: Text(AppConstants.backendLabel(backend)),
                            icon: Icon(_backendIcon(backend)),
                          ),
                        )
                        .toList(),
                    selected: {_selectedBackend},
                    onSelectionChanged: (newSelection) {
                      setState(() => _selectedBackend = newSelection.first);
                    },
                  ),
                  if (_selectedBackend == PreferredBackend.npu) ...[
                    const SizedBox(height: 6),
                    Text(
                      'NPU requer dispositivo compatível (Snapdragon / Google Tensor / Intel NPU) e variante NPU; em bundles genéricos, o runtime faz fallback automático para GPU/CPU.',
                      style: theme.textTheme.bodySmall?.copyWith(
                        color: theme.colorScheme.onSurfaceVariant,
                      ),
                    ),
                  ],
                  const SizedBox(height: 16),
                  Text(
                    'Janela de contexto (${AppConstants.formatTokenCount(_selectedMaxTokens)} • máx. ${AppConstants.formatTokenCount(modelSpec.maxContextTokens)}):',
                    style: const TextStyle(fontWeight: FontWeight.bold),
                  ),
                  const SizedBox(height: 8),
                  Wrap(
                    spacing: 6,
                    runSpacing: 6,
                    children: modelSpec.contextTokenPresets.map((tokens) {
                      return ChoiceChip(
                        label: Text(AppConstants.formatTokenCount(tokens)),
                        selected: _selectedMaxTokens == tokens,
                        onSelected: (_) {
                          setState(() => _selectedMaxTokens = tokens);
                        },
                      );
                    }).toList(),
                  ),
                  if (isMobile && _selectedMaxTokens >= 16384) ...[
                    const SizedBox(height: 6),
                    Text(
                      'Atenção: contextos de 16K–32K exigem mais memória RAM no Android.',
                      style: theme.textTheme.bodySmall?.copyWith(
                        color: theme.colorScheme.error,
                      ),
                    ),
                  ],
                  const SizedBox(height: 12),
                  if (modelSpec.supportsThinking)
                    SwitchListTile.adaptive(
                      dense: true,
                      contentPadding: EdgeInsets.zero,
                      secondary: const Icon(
                        Icons.psychology_alt_outlined,
                        color: Colors.teal,
                      ),
                      title: const Text(
                        'Modo Pensamento (Thinking)',
                        style: TextStyle(fontWeight: FontWeight.bold),
                      ),
                      subtitle: const Text(
                        'Mostra o raciocínio interno antes da resposta',
                        style: TextStyle(fontSize: 12),
                      ),
                      value: _selectedEnableThinking,
                      onChanged: (value) {
                        setState(() => _selectedEnableThinking = value);
                      },
                    )
                  else
                    Text(
                      '${modelSpec.displayName} não possui Modo Pensamento (Thinking).',
                      style: theme.textTheme.bodySmall?.copyWith(
                        color: theme.colorScheme.onSurfaceVariant,
                      ),
                    ),
                  const SizedBox(height: 12),
                  const Text(
                    'Estilo de Resposta:',
                    style: TextStyle(fontWeight: FontWeight.bold),
                  ),
                  const SizedBox(height: 8),
                  SegmentedButton<ResponseStyle>(
                    segments: const [
                      ButtonSegment<ResponseStyle>(
                        value: ResponseStyle.brief,
                        label: Text('Breve'),
                        icon: Icon(Icons.bolt),
                      ),
                      ButtonSegment<ResponseStyle>(
                        value: ResponseStyle.simple,
                        label: Text('Simples'),
                        icon: Icon(Icons.lightbulb_outline),
                      ),
                      ButtonSegment<ResponseStyle>(
                        value: ResponseStyle.detailed,
                        label: Text('Detalhada'),
                        icon: Icon(Icons.menu_book),
                      ),
                    ],
                    selected: {_selectedResponseStyle},
                    onSelectionChanged: (newSelection) {
                      setState(
                        () => _selectedResponseStyle = newSelection.first,
                      );
                    },
                  ),
                  const SizedBox(height: 8),
                  Text(
                    _selectedResponseStyle.description,
                    style: TextStyle(
                      fontSize: 12,
                      color: Colors.teal.shade800,
                      fontStyle: FontStyle.italic,
                    ),
                  ),
                  const SizedBox(height: 16),
                  Container(
                    padding: const EdgeInsets.all(12),
                    decoration: BoxDecoration(
                      color: Colors.teal.shade50,
                      borderRadius: BorderRadius.circular(8),
                      border: Border.all(color: Colors.teal.shade200),
                    ),
                    child: const Row(
                      children: [
                        Icon(Icons.sync, color: Colors.teal, size: 22),
                        SizedBox(width: 10),
                        Expanded(
                          child: Text(
                            'O contexto desta conversa será mantido. Alterar modelo, backend ou janela de contexto recarrega (ou baixa) o modelo; estilo e thinking aplicam instantaneamente.',
                            style: TextStyle(
                              fontSize: 12,
                              color: Colors.black87,
                            ),
                          ),
                        ),
                      ],
                    ),
                  ),
                ],
              );
            },
          ),
        ),
      ),
      actions: [
        TextButton(
          onPressed: () => Navigator.of(context).pop(),
          child: const Text('Cancelar'),
        ),
        FilledButton(
          onPressed: () {
            Navigator.of(context).pop();
            widget.controller.updateConversationSettings(
              modelId: _selectedModelId,
              preferredBackend: _selectedBackend,
              maxTokens: _selectedMaxTokens,
              enableThinking: _selectedEnableThinking,
              newResponseStyle: _selectedResponseStyle,
            );
          },
          child: const Text('Salvar'),
        ),
      ],
    );
  }
}

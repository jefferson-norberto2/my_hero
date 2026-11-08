import 'package:flutter/material.dart';
import '../../controller/chat_controller.dart';
import '../../data/adapter/ollama_model.dart';
import '../../data/adapter/response_style.dart';

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
  late String _selectedModel;
  late ResponseStyle _selectedResponseStyle;

  @override
  void initState() {
    super.initState();
    _selectedModel = widget.controller.selectedModelName.value;
    if (_selectedModel.isEmpty && widget.controller.availableModels.value.isNotEmpty) {
      _selectedModel = widget.controller.availableModels.value.first.name;
    }
    _selectedResponseStyle = widget.controller.responseStyle.value;
  }

  @override
  Widget build(BuildContext context) {
    final models = widget.controller.availableModels.value;

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
      content: SingleChildScrollView(
        child: Column(
          mainAxisSize: MainAxisSize.min,
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            const Text(
              'Modelo Ollama:',
              style: TextStyle(fontWeight: FontWeight.bold),
            ),
            const SizedBox(height: 8),
            if (models.isNotEmpty)
              DropdownButtonFormField<String>(
                isExpanded: true,
                initialValue: models.any((m) => m.name == _selectedModel)
                    ? _selectedModel
                    : models.first.name,
                decoration: const InputDecoration(
                  border: OutlineInputBorder(),
                  contentPadding: EdgeInsets.symmetric(horizontal: 12, vertical: 8),
                ),
                items: models.map((OllamaModel model) {
                  return DropdownMenuItem<String>(
                    value: model.name,
                    child: Text(
                      '${model.name}${model.parameterSize != null ? " (${model.parameterSize})" : ""}',
                      overflow: TextOverflow.ellipsis,
                    ),
                  );
                }).toList(),
                onChanged: (newVal) {
                  if (newVal != null) {
                    setState(() => _selectedModel = newVal);
                  }
                },
              )
            else
              const Text('Nenhum modelo disponível.'),
            const SizedBox(height: 16),
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
                setState(() => _selectedResponseStyle = newSelection.first);
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
                      'O contexto desta conversa e o novo modelo serão mantidos para as próximas perguntas.',
                      style: TextStyle(fontSize: 12, color: Colors.black87),
                    ),
                  ),
                ],
              ),
            ),
          ],
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
              modelName: _selectedModel,
              newResponseStyle: _selectedResponseStyle,
            );
          },
          child: const Text('Salvar'),
        ),
      ],
    );
  }
}

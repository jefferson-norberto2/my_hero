import 'package:flutter/material.dart';
import '../../controller/chat_controller.dart';
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
  late bool _selectedIs2B;
  late bool _selectedUseCpu;
  late ResponseStyle _selectedResponseStyle;

  @override
  void initState() {
    super.initState();
    _selectedIs2B = widget.controller.is2BModel.value;
    _selectedUseCpu = widget.controller.isCpu;
    _selectedResponseStyle = widget.controller.responseStyle.value;
  }

  @override
  Widget build(BuildContext context) {
    return AlertDialog(
      title: const Row(
        children: [
          Icon(Icons.tune, color: Colors.teal),
          SizedBox(width: 8),
          Text('Configurações da Conversa'),
        ],
      ),
      content: SingleChildScrollView(
        child: Column(
          mainAxisSize: MainAxisSize.min,
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            const Text(
              'Modelo Gemma:',
              style: TextStyle(fontWeight: FontWeight.bold),
            ),
            const SizedBox(height: 8),
            SegmentedButton<bool>(
              segments: const [
                ButtonSegment<bool>(
                  value: true,
                  label: Text('Gemma 2B'),
                  icon: Icon(Icons.flash_on),
                ),
                ButtonSegment<bool>(
                  value: false,
                  label: Text('Gemma 4B'),
                  icon: Icon(Icons.psychology),
                ),
              ],
              selected: {_selectedIs2B},
              onSelectionChanged: (newSelection) {
                setState(() => _selectedIs2B = newSelection.first);
              },
            ),
            const SizedBox(height: 16),
            const Text(
              'Backend de execução:',
              style: TextStyle(fontWeight: FontWeight.bold),
            ),
            const SizedBox(height: 8),
            SegmentedButton<bool>(
              segments: const [
                ButtonSegment<bool>(
                  value: true,
                  label: Text('CPU'),
                  icon: Icon(Icons.memory),
                ),
                ButtonSegment<bool>(
                  value: false,
                  label: Text('GPU'),
                  icon: Icon(Icons.speed),
                ),
              ],
              selected: {_selectedUseCpu},
              onSelectionChanged: (newSelection) {
                setState(() => _selectedUseCpu = newSelection.first);
              },
            ),
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
                      'O contexto desta conversa e o novo estilo serão mantidos para as próximas perguntas.',
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
              is2B: _selectedIs2B,
              useCpu: _selectedUseCpu,
              newResponseStyle: _selectedResponseStyle,
            );
          },
          child: const Text('Salvar'),
        ),
      ],
    );
  }
}

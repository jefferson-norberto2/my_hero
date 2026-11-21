import 'package:flutter/foundation.dart'
    show TargetPlatform, defaultTargetPlatform;
import 'package:flutter/material.dart';
import 'package:flutter_gemma/flutter_gemma.dart';
import '../../../../core/constants/app_constants.dart';
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
  late PreferredBackend _selectedBackend;
  late int _selectedMaxTokens;
  late bool _selectedEnableThinking;
  late ResponseStyle _selectedResponseStyle;

  @override
  void initState() {
    super.initState();
    _selectedIs2B = widget.controller.is2BModel.value;
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
    final modelSpec = AppConstants.modelSpec(is2B: _selectedIs2B);
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
      content: SingleChildScrollView(
        child: Column(
          mainAxisSize: MainAxisSize.min,
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            const Text(
              'Modelo Gemma 4:',
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
                setState(() {
                  _selectedIs2B = newSelection.first;
                  _selectedMaxTokens = AppConstants.clampMaxTokens(
                    _selectedMaxTokens,
                    is2B: _selectedIs2B,
                  );
                });
              },
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
                'NPU requer dispositivo compatível e modelo compilado para NPU; caso contrário, o fallback automático para GPU/CPU será aplicado.',
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
              children: AppConstants.contextTokenPresets.map((tokens) {
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
                      'O contexto desta conversa será mantido. Alterar modelo, backend ou janela de contexto recarrega o modelo; estilo e thinking aplicam instantaneamente.',
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

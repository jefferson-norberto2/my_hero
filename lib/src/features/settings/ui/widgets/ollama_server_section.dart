import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:my_hero/src/core/constants/app_constants.dart';

/// Card section displaying Ollama server address, optional hardware memory cap,
/// and an inline IP editor.
class OllamaServerSection extends StatelessWidget {
  final String serverUrl;
  final bool isConnected;
  final bool isEditing;
  final bool isTesting;
  final bool isApplying;
  final bool? testSuccess;
  final String? testMessage;
  final TextEditingController hostController;
  final TextEditingController memoryLimitController;
  final ValueChanged<String> onMemoryLimitChanged;
  final VoidCallback onToggleEditor;
  final VoidCallback onHostTextChanged;
  final VoidCallback onSelectLocalhost;
  final VoidCallback onTestConnection;
  final VoidCallback onApplyHost;

  const OllamaServerSection({
    super.key,
    required this.serverUrl,
    required this.isConnected,
    required this.isEditing,
    required this.isTesting,
    required this.isApplying,
    required this.testSuccess,
    required this.testMessage,
    required this.hostController,
    required this.memoryLimitController,
    required this.onMemoryLimitChanged,
    required this.onToggleEditor,
    required this.onHostTextChanged,
    required this.onSelectLocalhost,
    required this.onTestConnection,
    required this.onApplyHost,
  });

  static const List<int> _memoryPresetsGb = [16, 32, 64];

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
                Icon(
                  Icons.dns_rounded,
                  color: isConnected ? Colors.teal : Colors.orange,
                ),
                const SizedBox(width: 10),
                Expanded(
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text(
                        'Servidor: $serverUrl',
                        style: const TextStyle(
                          fontWeight: FontWeight.w600,
                          fontSize: 13,
                        ),
                        overflow: TextOverflow.ellipsis,
                      ),
                      const SizedBox(height: 2),
                      Text(
                        isConnected ? 'Conectado' : 'Desconectado',
                        style: TextStyle(
                          fontSize: 11,
                          fontWeight: FontWeight.w600,
                          color: isConnected
                              ? Colors.teal.shade700
                              : Colors.orange.shade800,
                        ),
                      ),
                    ],
                  ),
                ),
                TextButton.icon(
                  style: TextButton.styleFrom(
                    visualDensity: VisualDensity.compact,
                  ),
                  onPressed: isApplying ? null : onToggleEditor,
                  icon: Icon(
                    isEditing ? Icons.expand_less : Icons.edit_outlined,
                    size: 18,
                  ),
                  label: Text(isEditing ? 'Fechar' : 'Alterar IP'),
                ),
              ],
            ),
            const Divider(height: 24),
            TextField(
              key: const Key('server_memory_limit_field'),
              controller: memoryLimitController,
              decoration: InputDecoration(
                labelText: 'Memória disponível para modelos (GB - opcional)',
                hintText: 'Ex: 16, 32, 64',
                suffixText: 'GB',
                prefixIcon: const Icon(Icons.memory_rounded),
                suffixIcon: memoryLimitController.text.isNotEmpty
                    ? IconButton(
                        icon: const Icon(Icons.clear, size: 18),
                        tooltip: 'Limpar memória',
                        onPressed: () {
                          memoryLimitController.clear();
                          onMemoryLimitChanged('');
                        },
                      )
                    : null,
                border: OutlineInputBorder(
                  borderRadius: BorderRadius.circular(12),
                ),
                filled: true,
                fillColor: theme.colorScheme.surface,
              ),
              keyboardType:
                  const TextInputType.numberWithOptions(decimal: true),
              inputFormatters: [
                FilteringTextInputFormatter.allow(RegExp(r'[0-9.,]')),
              ],
              onChanged: onMemoryLimitChanged,
            ),
            const SizedBox(height: 8),
            Wrap(
              spacing: 8,
              runSpacing: 6,
              children: _memoryPresetsGb.map((presetGb) {
                return ActionChip(
                  label: Text('$presetGb GB'),
                  onPressed: () {
                    final value = presetGb.toString();
                    memoryLimitController.text = value;
                    onMemoryLimitChanged(value);
                  },
                );
              }).toList(),
            ),
            if (isEditing) ...[
              const Divider(height: 24),
              const Text(
                'Digite o IP ou endereço do computador onde o Ollama está rodando na sua rede local:',
                style: TextStyle(fontSize: 13),
              ),
              const SizedBox(height: 12),
              TextField(
                key: const Key('server_host_field'),
                controller: hostController,
                decoration: InputDecoration(
                  labelText: 'Endereço / IP do Ollama',
                  hintText: 'Ex: 192.168.1.50 ou 192.168.1.50:11434',
                  prefixIcon: const Icon(Icons.lan_outlined),
                  suffixIcon: IconButton(
                    icon: const Icon(Icons.clear, size: 18),
                    tooltip: 'Limpar',
                    onPressed: () {
                      hostController.clear();
                      onHostTextChanged();
                    },
                  ),
                  border: OutlineInputBorder(
                    borderRadius: BorderRadius.circular(12),
                  ),
                  filled: true,
                  fillColor: theme.colorScheme.surface,
                ),
                keyboardType: TextInputType.url,
                onChanged: (_) => onHostTextChanged(),
              ),
              const SizedBox(height: 8),
              Wrap(
                spacing: 8,
                children: [
                  ActionChip(
                    avatar: const Icon(Icons.computer, size: 16),
                    label: const Text('Localhost (padrão)'),
                    onPressed: () {
                      hostController.text = AppConstants.ollamaBaseUrl;
                      onSelectLocalhost();
                    },
                  ),
                ],
              ),
              const SizedBox(height: 12),
              Container(
                padding: const EdgeInsets.all(10),
                decoration: BoxDecoration(
                  color: theme.colorScheme.primaryContainer.withValues(alpha: 0.35),
                  borderRadius: BorderRadius.circular(12),
                  border: Border.all(
                    color: theme.colorScheme.primary.withValues(alpha: 0.3),
                  ),
                ),
                child: const Row(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Icon(Icons.lightbulb_outline, size: 20, color: Colors.teal),
                    SizedBox(width: 8),
                    Expanded(
                      child: Text(
                        'Dica para conexão de rede: Na máquina onde o Ollama roda, certifique-se de iniciar o serviço com `OLLAMA_HOST=0.0.0.0 ollama serve` para aceitar conexões da rede local.',
                        style: TextStyle(fontSize: 11),
                      ),
                    ),
                  ],
                ),
              ),
              if (testMessage != null) ...[
                const SizedBox(height: 12),
                Container(
                  padding: const EdgeInsets.all(10),
                  decoration: BoxDecoration(
                    color: testSuccess == true
                        ? Colors.teal.shade50
                        : Colors.red.shade50,
                    borderRadius: BorderRadius.circular(12),
                    border: Border.all(
                      color: testSuccess == true
                          ? Colors.teal.shade300
                          : Colors.red.shade300,
                    ),
                  ),
                  child: Row(
                    children: [
                      Icon(
                        testSuccess == true
                            ? Icons.check_circle_outline
                            : Icons.error_outline,
                        color: testSuccess == true
                            ? Colors.teal.shade800
                            : Colors.red.shade800,
                        size: 20,
                      ),
                      const SizedBox(width: 8),
                      Expanded(
                        child: Text(
                          testMessage!,
                          style: TextStyle(
                            fontSize: 12,
                            color: testSuccess == true
                                ? Colors.teal.shade900
                                : Colors.red.shade900,
                            fontWeight: FontWeight.w500,
                          ),
                        ),
                      ),
                    ],
                  ),
                ),
              ],
              const SizedBox(height: 12),
              Wrap(
                spacing: 8,
                runSpacing: 8,
                alignment: WrapAlignment.end,
                children: [
                  OutlinedButton.icon(
                    onPressed: isTesting || isApplying ? null : onTestConnection,
                    icon: isTesting
                        ? const SizedBox(
                            width: 14,
                            height: 14,
                            child: CircularProgressIndicator(strokeWidth: 2),
                          )
                        : const Icon(Icons.network_ping, size: 18),
                    label: const Text('Testar Conexão'),
                  ),
                  FilledButton.icon(
                    onPressed: isApplying ? null : onApplyHost,
                    icon: isApplying
                        ? const SizedBox(
                            width: 14,
                            height: 14,
                            child: CircularProgressIndicator(
                              strokeWidth: 2,
                              color: Colors.white,
                            ),
                          )
                        : const Icon(Icons.check, size: 18),
                    label: const Text('Aplicar IP'),
                  ),
                ],
              ),
            ],
          ],
        ),
      ),
    );
  }
}

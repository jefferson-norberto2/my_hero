import 'package:flutter/material.dart';
import '../../controller/chat_controller.dart';
import '../../../../core/utils/url_utils.dart';

class OllamaServerConfigDialog extends StatefulWidget {
  final ChatController controller;

  const OllamaServerConfigDialog({super.key, required this.controller});

  static Future<void> show(BuildContext context, ChatController controller) {
    return showDialog(
      context: context,
      builder: (_) => OllamaServerConfigDialog(controller: controller),
    );
  }

  @override
  State<OllamaServerConfigDialog> createState() => _OllamaServerConfigDialogState();
}

class _OllamaServerConfigDialogState extends State<OllamaServerConfigDialog> {
  late final TextEditingController _hostController;
  bool _isTesting = false;
  bool _isSaving = false;
  bool? _testSuccess;
  String? _testMessage;

  @override
  void initState() {
    super.initState();
    _hostController = TextEditingController(
      text: widget.controller.ollamaBaseUrl.value,
    );
  }

  @override
  void dispose() {
    _hostController.dispose();
    super.dispose();
  }

  Future<void> _testConnection() async {
    final input = _hostController.text.trim();
    if (input.isEmpty) return;

    setState(() {
      _isTesting = true;
      _testSuccess = null;
      _testMessage = null;
    });

    try {
      final normalized = normalizeOllamaUrl(input);
      final isHealthy = await widget.controller.testOllamaConnection(normalized);

      if (!mounted) return;
      setState(() {
        _isTesting = false;
        _testSuccess = isHealthy;
        _testMessage = isHealthy
            ? 'Conectado com sucesso em $normalized!'
            : 'Não foi possível conectar em $normalized. Verifique o IP, porta e firewall.';
      });
    } catch (e) {
      if (!mounted) return;
      setState(() {
        _isTesting = false;
        _testSuccess = false;
        _testMessage = 'Erro de conexão: $e';
      });
    }
  }

  Future<void> _saveAndConnect() async {
    final input = _hostController.text.trim();
    setState(() {
      _isSaving = true;
    });

    final normalized = normalizeOllamaUrl(input);
    final connected = await widget.controller.updateOllamaHost(normalized);

    if (!mounted) return;
    Navigator.of(context).pop();

    ScaffoldMessenger.of(context).showSnackBar(
      SnackBar(
        content: Text(
          connected
              ? 'Conectado ao Ollama em $normalized'
              : 'Endereço atualizado para $normalized, mas o Ollama não respondeu.',
        ),
        backgroundColor: connected ? Colors.teal : Colors.orange.shade800,
        behavior: SnackBarBehavior.floating,
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);

    return AlertDialog(
      title: const Row(
        children: [
          Icon(Icons.dns_rounded, color: Colors.teal),
          SizedBox(width: 10),
          Expanded(
            child: Text(
              'Servidor Ollama na Rede',
              style: TextStyle(fontSize: 18),
            ),
          ),
        ],
      ),
      content: SingleChildScrollView(
        child: Column(
          mainAxisSize: MainAxisSize.min,
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            const Text(
              'Digite o IP ou endereço do computador onde o Ollama está rodando na sua rede local:',
              style: TextStyle(fontSize: 13),
            ),
            const SizedBox(height: 12),
            TextField(
              controller: _hostController,
              decoration: InputDecoration(
                labelText: 'Endereço / IP do Ollama',
                hintText: 'Ex: 192.168.1.50 ou 192.168.1.50:11434',
                prefixIcon: const Icon(Icons.lan_outlined),
                suffixIcon: IconButton(
                  icon: const Icon(Icons.clear, size: 18),
                  tooltip: 'Limpar',
                  onPressed: () => _hostController.clear(),
                ),
                border: const OutlineInputBorder(),
              ),
              keyboardType: TextInputType.url,
              onChanged: (_) {
                if (_testSuccess != null) {
                  setState(() {
                    _testSuccess = null;
                    _testMessage = null;
                  });
                }
              },
            ),
            const SizedBox(height: 8),
            Wrap(
              spacing: 8,
              children: [
                ActionChip(
                  avatar: const Icon(Icons.computer, size: 16),
                  label: const Text('Localhost (padrão)'),
                  onPressed: () {
                    _hostController.text = 'http://localhost:11434';
                    setState(() {
                      _testSuccess = null;
                      _testMessage = null;
                    });
                  },
                ),
              ],
            ),
            const SizedBox(height: 12),
            // Cartão explicativo com dica do OLLAMA_HOST
            Container(
              padding: const EdgeInsets.all(10),
              decoration: BoxDecoration(
                color: theme.colorScheme.primaryContainer.withValues(alpha: 0.35),
                borderRadius: BorderRadius.circular(8),
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
            if (_testMessage != null) ...[
              const SizedBox(height: 12),
              Container(
                padding: const EdgeInsets.all(10),
                decoration: BoxDecoration(
                  color: _testSuccess == true
                      ? Colors.teal.shade50
                      : Colors.red.shade50,
                  borderRadius: BorderRadius.circular(8),
                  border: Border.all(
                    color: _testSuccess == true
                        ? Colors.teal.shade300
                        : Colors.red.shade300,
                  ),
                ),
                child: Row(
                  children: [
                    Icon(
                      _testSuccess == true
                          ? Icons.check_circle_outline
                          : Icons.error_outline,
                      color: _testSuccess == true
                          ? Colors.teal.shade800
                          : Colors.red.shade800,
                      size: 20,
                    ),
                    const SizedBox(width: 8),
                    Expanded(
                      child: Text(
                        _testMessage!,
                        style: TextStyle(
                          fontSize: 12,
                          color: _testSuccess == true
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
          ],
        ),
      ),
      actions: [
        OutlinedButton.icon(
          onPressed: _isTesting || _isSaving ? null : _testConnection,
          icon: _isTesting
              ? const SizedBox(
                  width: 14,
                  height: 14,
                  child: CircularProgressIndicator(strokeWidth: 2),
                )
              : const Icon(Icons.network_ping, size: 18),
          label: const Text('Testar Conexão'),
        ),
        TextButton(
          onPressed: _isSaving ? null : () => Navigator.of(context).pop(),
          child: const Text('Cancelar'),
        ),
        FilledButton.icon(
          onPressed: _isSaving ? null : _saveAndConnect,
          icon: _isSaving
              ? const SizedBox(
                  width: 14,
                  height: 14,
                  child: CircularProgressIndicator(
                    strokeWidth: 2,
                    color: Colors.white,
                  ),
                )
              : const Icon(Icons.check, size: 18),
          label: const Text('Salvar e Conectar'),
        ),
      ],
    );
  }
}

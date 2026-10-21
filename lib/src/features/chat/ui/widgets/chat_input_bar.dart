import 'package:flutter/material.dart';
import 'package:signals_flutter/signals_flutter.dart';
import '../../controller/chat_controller.dart';

class ChatInputBar extends StatefulWidget {
  final ChatController controller;

  const ChatInputBar({super.key, required this.controller});

  @override
  State<ChatInputBar> createState() => _ChatInputBarState();
}

class _ChatInputBarState extends State<ChatInputBar> {
  final TextEditingController _textController = TextEditingController();

  @override
  void dispose() {
    _textController.dispose();
    super.dispose();
  }

  void _handleSend() {
    final text = _textController.text;
    final hasAttachedImage = widget.controller.attachedImage.value != null;
    if (text.trim().isNotEmpty || hasAttachedImage) {
      widget.controller.sendMessage(text);
      _textController.clear();
    }
  }

  void _showMediaPickerOptions() {
    showModalBottomSheet(
      context: context,
      showDragHandle: true,
      builder: (ctx) => SafeArea(
        child: Padding(
          padding: const EdgeInsets.only(bottom: 8.0),
          child: Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              ListTile(
                leading: const Icon(Icons.photo_library_rounded, color: Colors.teal),
                title: const Text('Galeria de fotos'),
                subtitle: const Text('Selecione uma imagem salva no dispositivo'),
                onTap: () {
                  Navigator.of(ctx).pop();
                  widget.controller.pickImage(fromCamera: false);
                },
              ),
              ListTile(
                leading: const Icon(Icons.camera_alt_rounded, color: Colors.teal),
                title: const Text('Câmera'),
                subtitle: const Text('Tire uma foto para análise imediata'),
                onTap: () {
                  Navigator.of(ctx).pop();
                  widget.controller.pickImage(fromCamera: true);
                },
              ),
            ],
          ),
        ),
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    return SignalBuilder(
      builder: (context) {
        final isModelLoaded = widget.controller.isModelLoaded.value;
        final isGenerating = widget.controller.isGenerating.value;
        final attachedImage = widget.controller.attachedImage.value;

        final canType = isModelLoaded && !isGenerating;

        return Column(
          mainAxisSize: MainAxisSize.min,
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            if (attachedImage != null)
              Padding(
                padding: const EdgeInsets.only(bottom: 8.0),
                child: Stack(
                  clipBehavior: Clip.none,
                  children: [
                    Container(
                      padding: const EdgeInsets.all(4.0),
                      decoration: BoxDecoration(
                        color: Theme.of(context).colorScheme.surfaceContainerHighest,
                        borderRadius: BorderRadius.circular(12),
                        border: Border.all(
                          color: Colors.teal.withValues(alpha: 0.5),
                          width: 1.5,
                        ),
                      ),
                      child: Row(
                        mainAxisSize: MainAxisSize.min,
                        children: [
                          ClipRRect(
                            borderRadius: BorderRadius.circular(8),
                            child: Image.memory(
                              attachedImage,
                              width: 54,
                              height: 54,
                              fit: BoxFit.cover,
                            ),
                          ),
                          const SizedBox(width: 10),
                          Column(
                            crossAxisAlignment: CrossAxisAlignment.start,
                            mainAxisSize: MainAxisSize.min,
                            children: [
                              Text(
                                'Imagem anexada',
                                style: Theme.of(context).textTheme.labelMedium?.copyWith(
                                      fontWeight: FontWeight.bold,
                                      color: Colors.teal,
                                    ),
                              ),
                              Text(
                                'Pronta para análise multimodal',
                                style: Theme.of(context).textTheme.bodySmall?.copyWith(
                                      color: Theme.of(context).colorScheme.outline,
                                    ),
                              ),
                            ],
                          ),
                          const SizedBox(width: 8),
                        ],
                      ),
                    ),
                    Positioned(
                      top: -6,
                      right: -6,
                      child: GestureDetector(
                        onTap: widget.controller.clearAttachedImage,
                        child: Container(
                          decoration: BoxDecoration(
                            color: Theme.of(context).colorScheme.error,
                            shape: BoxShape.circle,
                          ),
                          padding: const EdgeInsets.all(4),
                          child: const Icon(
                            Icons.close_rounded,
                            size: 14,
                            color: Colors.white,
                          ),
                        ),
                      ),
                    ),
                  ],
                ),
              ),
            Row(
              children: [
                Expanded(
                  child: TextField(
                    controller: _textController,
                    enabled: canType,
                    decoration: InputDecoration(
                      hintText: attachedImage != null
                          ? 'Faça uma pergunta sobre a imagem ou envie direto...'
                          : 'Digite sua mensagem...',
                      border: const OutlineInputBorder(),
                    ),
                    onSubmitted: (_) => _handleSend(),
                  ),
                ),
                IconButton(
                  onPressed: canType ? _showMediaPickerOptions : null,
                  color: Colors.teal,
                  tooltip: 'Anexar imagem (Galeria ou Câmera)',
                  icon: const Icon(Icons.add_photo_alternate_rounded),
                ),
                const SizedBox(width: 8),
                IconButton(
                  icon: Icon(
                    !isGenerating ? Icons.send : Icons.stop_circle_rounded,
                  ),
                  tooltip: !isGenerating ? 'Enviar mensagem' : 'Parar geração',
                  color: !isGenerating ? Colors.teal : Colors.red,
                  onPressed: !isModelLoaded
                      ? null
                      : (!isGenerating ? _handleSend : widget.controller.stopGeneration),
                ),
              ],
            ),
          ],
        );
      },
    );
  }
}

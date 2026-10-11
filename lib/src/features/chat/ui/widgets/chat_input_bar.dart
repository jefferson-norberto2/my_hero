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
    if (text.trim().isNotEmpty) {
      widget.controller.sendMessage(text);
      _textController.clear();
    }
  }

  void _handlePickImage() {
    widget.controller.pickAndSendImage(_textController.text);
    _textController.clear();
  }

  @override
  Widget build(BuildContext context) {
    return SignalBuilder(
      builder: (context) {
        final isModelLoaded = widget.controller.isModelLoaded.value;
        final isGenerating = widget.controller.isGenerating.value;

        final canType = isModelLoaded && !isGenerating;

        return Row(
          children: [
            Expanded(
              child: TextField(
                controller: _textController,
                enabled: canType,
                decoration: const InputDecoration(
                  hintText: 'Digite sua mensagem...',
                  border: OutlineInputBorder(),
                ),
                onSubmitted: (_) => _handleSend(),
              ),
            ),
            IconButton(
              onPressed: canType ? _handlePickImage : null,
              color: Colors.teal,
              icon: const Icon(Icons.add_photo_alternate_rounded),
            ),
            const SizedBox(width: 8),
            IconButton(
              icon: Icon(
                !isGenerating ? Icons.send : Icons.stop_circle_rounded,
              ),
              color: !isGenerating ? Colors.teal : Colors.red,
              onPressed: !isModelLoaded
                  ? null
                  : (!isGenerating ? _handleSend : widget.controller.stopGeneration),
            ),
          ],
        );
      },
    );
  }
}

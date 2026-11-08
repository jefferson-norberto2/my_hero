import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:signals_flutter/signals_flutter.dart';
import '../../controller/chat_controller.dart';
import '../../data/adapter/chat_attachment.dart';

class ChatInputBar extends StatefulWidget {
  final ChatController controller;

  const ChatInputBar({super.key, required this.controller});

  @override
  State<ChatInputBar> createState() => _ChatInputBarState();
}

class _ChatInputBarState extends State<ChatInputBar> {
  final TextEditingController _textController = TextEditingController();
  final FocusNode _focusNode = FocusNode();

  @override
  void dispose() {
    _textController.dispose();
    _focusNode.dispose();
    super.dispose();
  }

  void _handleSend() {
    final text = _textController.text;
    final hasAttachment = widget.controller.attachedAttachment.value != null;
    if (text.trim().isNotEmpty || hasAttachment) {
      widget.controller.sendMessage(text);
      _textController.clear();
    }
  }

  void _showAttachmentOptions() {
    showModalBottomSheet(
      context: context,
      showDragHandle: true,
      builder: (ctx) => SafeArea(
        child: Padding(
          padding: const EdgeInsets.symmetric(horizontal: 8.0, vertical: 4.0),
          child: Column(
            mainAxisSize: MainAxisSize.min,
            crossAxisAlignment: CrossAxisAlignment.stretch,
            children: [
              const Padding(
                padding: EdgeInsets.symmetric(horizontal: 16.0, vertical: 6.0),
                child: Text(
                  'Adicionar Anexo',
                  style: TextStyle(fontSize: 16, fontWeight: FontWeight.bold),
                ),
              ),
              ListTile(
                leading: Container(
                  padding: const EdgeInsets.all(8),
                  decoration: BoxDecoration(
                    color: Colors.teal.withValues(alpha: 0.15),
                    borderRadius: BorderRadius.circular(8),
                  ),
                  child: const Icon(Icons.image_rounded, color: Colors.teal),
                ),
                title: const Text('Imagem'),
                subtitle: const Text('Foto da galeria ou da câmera'),
                trailing: const Icon(Icons.chevron_right_rounded),
                onTap: () {
                  Navigator.of(ctx).pop();
                  _showImagePickerSubmenu();
                },
              ),
              ListTile(
                leading: Container(
                  padding: const EdgeInsets.all(8),
                  decoration: BoxDecoration(
                    color: Colors.amber.withValues(alpha: 0.15),
                    borderRadius: BorderRadius.circular(8),
                  ),
                  child: Icon(Icons.audiotrack_rounded, color: Colors.amber.shade800),
                ),
                title: const Text('Áudio'),
                subtitle: const Text('Arquivo de áudio (.mp3, .wav, etc.)'),
                onTap: () {
                  Navigator.of(ctx).pop();
                  widget.controller.pickAudio();
                },
              ),
              ListTile(
                leading: Container(
                  padding: const EdgeInsets.all(8),
                  decoration: BoxDecoration(
                    color: Colors.blue.withValues(alpha: 0.15),
                    borderRadius: BorderRadius.circular(8),
                  ),
                  child: const Icon(Icons.description_rounded, color: Colors.blue),
                ),
                title: const Text('Documento ou Código'),
                subtitle: const Text('Código-fonte, texto, PDF (.dart, .py, .txt, etc.)'),
                onTap: () {
                  Navigator.of(ctx).pop();
                  widget.controller.pickDocument();
                },
              ),
              const SizedBox(height: 8),
            ],
          ),
        ),
      ),
    );
  }

  void _showImagePickerSubmenu() {
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
                subtitle: const Text('Selecione uma imagem do computador'),
                onTap: () {
                  Navigator.of(ctx).pop();
                  widget.controller.pickImage(fromCamera: false);
                },
              ),
              ListTile(
                leading: const Icon(Icons.camera_alt_rounded, color: Colors.teal),
                title: const Text('Câmera'),
                subtitle: const Text('Tirar foto agora'),
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

  Widget _buildAttachmentPreview(BuildContext context, ChatAttachment attachment) {
    final theme = Theme.of(context);

    Widget leadingWidget;
    String subtitleText;

    switch (attachment.type) {
      case AttachmentType.image:
        leadingWidget = ClipRRect(
          borderRadius: BorderRadius.circular(8),
          child: Image.memory(
            attachment.bytes,
            width: 50,
            height: 50,
            fit: BoxFit.cover,
          ),
        );
        subtitleText = 'Imagem (${attachment.sizeFormatted}) • Análise multimodal';
        break;
      case AttachmentType.audio:
        leadingWidget = Container(
          width: 50,
          height: 50,
          decoration: BoxDecoration(
            color: Colors.amber.shade100,
            borderRadius: BorderRadius.circular(8),
          ),
          child: Icon(Icons.audiotrack_rounded, color: Colors.amber.shade800, size: 28),
        );
        subtitleText = 'Áudio (${attachment.sizeFormatted})';
        break;
      case AttachmentType.document:
        leadingWidget = Container(
          width: 50,
          height: 50,
          decoration: BoxDecoration(
            color: Colors.blue.shade100,
            borderRadius: BorderRadius.circular(8),
          ),
          child: const Icon(Icons.description_rounded, color: Colors.blue, size: 28),
        );
        subtitleText = attachment.textContent != null
            ? 'Documento/Código (${attachment.sizeFormatted}) • Texto extraído'
            : 'Documento (${attachment.sizeFormatted})';
        break;
    }

    return Padding(
      padding: const EdgeInsets.only(bottom: 8.0),
      child: Stack(
        clipBehavior: Clip.none,
        children: [
          Container(
            padding: const EdgeInsets.all(6.0),
            decoration: BoxDecoration(
              color: theme.colorScheme.surfaceContainerHighest,
              borderRadius: BorderRadius.circular(12),
              border: Border.all(
                color: Colors.teal.withValues(alpha: 0.5),
                width: 1.5,
              ),
            ),
            child: Row(
              mainAxisSize: MainAxisSize.min,
              children: [
                leadingWidget,
                const SizedBox(width: 10),
                Flexible(
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    mainAxisSize: MainAxisSize.min,
                    children: [
                      Text(
                        attachment.name,
                        maxLines: 1,
                        overflow: TextOverflow.ellipsis,
                        style: theme.textTheme.labelMedium?.copyWith(
                          fontWeight: FontWeight.bold,
                          color: Colors.teal.shade800,
                        ),
                      ),
                      Text(
                        subtitleText,
                        maxLines: 1,
                        overflow: TextOverflow.ellipsis,
                        style: theme.textTheme.bodySmall?.copyWith(
                          color: theme.colorScheme.outline,
                          fontSize: 11,
                        ),
                      ),
                    ],
                  ),
                ),
                const SizedBox(width: 8),
              ],
            ),
          ),
          Positioned(
            top: -6,
            right: -6,
            child: GestureDetector(
              onTap: widget.controller.clearAttachment,
              child: Container(
                decoration: BoxDecoration(
                  color: theme.colorScheme.error,
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
    );
  }

  @override
  Widget build(BuildContext context) {
    return SignalBuilder(
      builder: (context) {
        final isModelLoaded = widget.controller.isModelLoaded.value;
        final isGenerating = widget.controller.isGenerating.value;
        final attachment = widget.controller.attachedAttachment.value;

        final canType = isModelLoaded && !isGenerating;

        return Column(
          mainAxisSize: MainAxisSize.min,
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            if (attachment != null) _buildAttachmentPreview(context, attachment),
            Row(
              crossAxisAlignment: CrossAxisAlignment.end,
              children: [
                Expanded(
                  child: Focus(
                    onKeyEvent: (node, event) {
                      if (event is KeyDownEvent &&
                          (HardwareKeyboard.instance.isControlPressed ||
                              HardwareKeyboard.instance.isMetaPressed) &&
                          event.logicalKey == LogicalKeyboardKey.enter) {
                        _handleSend();
                        return KeyEventResult.handled;
                      }
                      return KeyEventResult.ignored;
                    },
                    child: TextField(
                      focusNode: _focusNode,
                      controller: _textController,
                      enabled: canType,
                      minLines: 2,
                      maxLines: 10,
                      keyboardType: TextInputType.multiline,
                      textInputAction: TextInputAction.newline,
                      style: const TextStyle(fontSize: 14),
                      decoration: InputDecoration(
                        hintText: attachment != null
                            ? 'Faça uma pergunta sobre o anexo... (Ctrl+Enter para enviar)'
                            : 'Digite sua mensagem ou cole código aqui...\n(Enter cria nova linha, Ctrl+Enter envia)',
                        hintMaxLines: 2,
                        contentPadding: const EdgeInsets.symmetric(horizontal: 16, vertical: 12),
                        border: OutlineInputBorder(
                          borderRadius: BorderRadius.circular(16),
                        ),
                      ),
                    ),
                  ),
                ),
                const SizedBox(width: 6),
                Padding(
                  padding: const EdgeInsets.only(bottom: 4.0),
                  child: Row(
                    mainAxisSize: MainAxisSize.min,
                    children: [
                      IconButton(
                        onPressed: canType ? _showAttachmentOptions : null,
                        color: Colors.teal,
                        tooltip: 'Anexar arquivo (Imagem, Áudio ou Documento)',
                        icon: const Icon(Icons.attach_file_rounded),
                      ),
                      IconButton(
                        icon: Icon(
                          !isGenerating ? Icons.send : Icons.stop_circle_rounded,
                        ),
                        tooltip: !isGenerating ? 'Enviar mensagem (Ctrl+Enter)' : 'Parar geração',
                        color: !isGenerating ? Colors.teal : Colors.red,
                        onPressed: !isModelLoaded
                            ? null
                            : (!isGenerating ? _handleSend : widget.controller.stopGeneration),
                      ),
                    ],
                  ),
                ),
              ],
            ),
          ],
        );
      },
    );
  }
}

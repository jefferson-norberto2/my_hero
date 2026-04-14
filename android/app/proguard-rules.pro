# Mantém todas as classes do MediaPipe intactas
-keep class com.google.mediapipe.** { *; }
-dontwarn com.google.mediapipe.**

# Mantém as classes de protobuf (usadas pelos modelos locais)
-keep class com.google.protobuf.** { *; }
-dontwarn com.google.protobuf.**

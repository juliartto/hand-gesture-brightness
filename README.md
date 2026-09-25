# hand-gesture-brightness

Controla o brilho da imagem da webcam com o gesto de pinça (polegar e indicador), usando OpenCV.

Trabalho 2 — Segmentação · Engenharia de Computação · UTFPR 

## Instalação

```bash
pip install opencv-python numpy
python controle_gestual.py
```

## Uso

1. Deixe o retângulo vazio e aperte **ESPAÇO** (captura o fundo).
2. Mostre a mão: polegar para o lado, indicador para cima.
3. Aperte **C** e abaixe e levante o indicador por 3 s (calibração).
4. Abaixe o indicador para escurecer e levante para clarear.

`+` / `-` ajustam o limiar · `Q` sai

## Como funciona

1. Filtro gaussiano e subtração do fundo.
2. Binarização com `cv2.threshold`.
3. Contorno da mão com `cv2.findContours`.
4. Polegar e indicador com `cv2.convexHull` e `cv2.convexityDefects`.
5. A distância entre os dois dedos vira o brilho (0 a 100 %).

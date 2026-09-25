# BIBLIOTECAS ------------------------------------------------------------
from collections import deque
from time import perf_counter

import cv2
import numpy as np
# ------------------------------------------------------------------------

JANELA = "Trabalho 2 - Controle gestual de brilho"
ROI = (320, 50, 620, 460)          # retângulo da mão: x1, y1, x2, y2
QUADROS_FUNDO = 30                 # quadros usados no modelo do fundo (~1 s)
TEMPO_CALIBRACAO = 3.0             # duração da calibração do gesto (s)
KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
VERDE, AZUL, AMARELO = (80, 220, 80), (255, 180, 40), (40, 220, 255)  # BGR
VERMELHO, BRANCO, imagem_cinza = (60, 60, 255), (235, 235, 235), (140, 140, 140)

# PRÉ-PROCESSAMENTO ------------------------------------------------------
def preprocessamento(regiao_mao):
    return cv2.GaussianBlur(regiao_mao, (5, 5), 0)

def binarizacao(imagem_processada, fundo, limiar):
    # se o pixel for igual ao do fundo, diferenca retorna 0, se for diferente retorna um valor maior, filtramos com o limiar no threshold
    diferenca = cv2.absdiff(imagem_processada, fundo)
    diferenca = cv2.cvtColor(diferenca, cv2.COLOR_BGR2HSV)[:, :, 2] # canal V (brilho)
    _, mascara = cv2.threshold(diferenca, limiar, 255, cv2.THRESH_BINARY)

    # limpa a mask
    mascara = cv2.morphologyEx(mascara, cv2.MORPH_OPEN, KERNEL)
    return cv2.morphologyEx(mascara, cv2.MORPH_CLOSE, KERNEL)
# ------------------------------------------------------------------------

# CALIBRAÇÃO DO FUNDO ----------------------------------------------------
def calibrar_fundo(quadros):
    pilha = np.stack(quadros) # junta os 30 quadros em uma pilha 3D (altura, largura, número de quadros)
    fundo = np.median(pilha, axis=0) # tira a mediana de cada pixel ao longo do eixo dos quadros, resultando em uma imagem só
    print(f"Fundo capturado.")
    return fundo.astype(np.uint8)
# ------------------------------------------------------------------------

# ANÁLISE DA MÃO ---------------------------------------------------------
def analisar_mao(mascara):
    # RETR_CCOMP separa contornos externos (mão) e internos (furos).
    contornos, hierarquia = cv2.findContours(mascara, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    # se não tiver contornos (só fundo), retorna None
    if not contornos:
        return None
    externos = [i for i, h in enumerate(hierarquia[0]) if h[3] == -1]
    indice = max(externos, key=lambda i: cv2.contourArea(contornos[i]))
    contorno = contornos[indice]

    # Descarta ruído e mão cortada pelo retângulo (só o punho sai por baixo).
    x, y, w, h = cv2.boundingRect(contorno)
    if (cv2.contourArea(contorno) < 2000 or x <= 1 or y <= 1
            or x + w >= mascara.shape[1] - 1):
        return None

    # Palma: maior círculo inscrito na mão (máximo da transformada de
    # distância). Os furos ficam vazios para o anel do OK não virar "palma".
    cheia = np.zeros_like(mascara)
    cv2.drawContours(cheia, [contorno], -1, 255, cv2.FILLED)
    distancias = cv2.distanceTransform(cv2.bitwise_and(cheia, mascara), cv2.DIST_L2, 5)
    _, raio, _, centro = cv2.minMaxLoc(distancias)
    if raio < 15:
        return None
    c = np.array(centro, dtype=float)
    mao = dict(contorno=contorno, casca=cv2.convexHull(contorno), centro=centro,
               raio=raio, furo=None, pinca=None, abertura=None)

    # Pinça fechada (sinal de OK): o indicador desce até o polegar e o vão
    # entre eles vira um furo (contorno filho) fora da palma e acima do punho.
    for i, h in enumerate(hierarquia[0]):
        m = cv2.moments(contornos[i])
        if h[3] != indice or m["m00"] < 0.1 * raio**2:  # ignora furos pequenos
            continue
        centro_furo = np.array([m["m10"], m["m01"]]) / m["m00"]
        if centro_furo[1] < c[1] + 0.4 * raio and np.linalg.norm(centro_furo - c) > 0.7 * raio:
            mao.update(furo=contornos[i], abertura=0.0)
            return mao

    # Pinça aberta: o vão entre polegar e indicador é um defeito de
    # convexidade, que liga duas pontas da casca passando pelo ponto mais fundo.
    try:
        defeitos = cv2.convexityDefects(contorno, cv2.convexHull(contorno, returnPoints=False))
    except cv2.error:  # contornos muito irregulares: ignora o quadro
        defeitos = None
    if defeitos is None:
        return mao

    # O polegar fica parado ao lado da palma e o indicador desce até ele. A
    # pinça é o defeito que contém a ponta do polegar: entre as pontas de dedo
    # (longe da palma) que ficam ao lado da palma, e não acima dela, é a mais
    # baixa. Não se supõe qual ponta está mais alta, pois o indicador desce.
    mais_baixa = -1
    for inicio, fim, meio, profundidade in defeitos[:, 0]:
        p1, p2, vale = contorno[[inicio, fim, meio], 0]
        baixa = max(p1, p2, key=lambda p: p[1])  # candidata a ponta do polegar
        dx, dy = baixa - c
        valido = (
            profundidade / 256 > 0.25 * raio        # vale real (ponto fixo x256)
            and abs(dx) > abs(dy)                   # ponta ao lado da palma
            and min(np.linalg.norm(p1 - c), np.linalg.norm(p2 - c)) > 1.5 * raio  # pontas de dedos
        )
        if valido and baixa[1] > mais_baixa:
            mais_baixa = baixa[1]
            mao.update(pinca=[tuple(map(int, p)) for p in (p1, p2, vale)],
                       abertura=float(np.linalg.norm(p1 - p2)) / raio)
    return mao
# ------------------------------------------------------------------------

def calibrar_gesto(amostras):
    """Converte as aberturas medidas na calibração nos limites 0 % e 100 %.

    Usa só a pinça aberta, pois a fechada (abertura 0) já é 0 %. Assim, 0 % é
    a menor abertura antes do toque e o nível não salta ao encostar os dedos.
    Os percentis 5 e 95 descartam detecções erradas isoladas nos extremos.
    """
    abertas = [a for a in amostras if a > 0]
    if len(abertas) < 15:
        print("Calibracao falhou: pinca pouco detectada. Aperte C de novo.")
        return None
    minimo, maximo = (float(v) for v in np.percentile(abertas, [5, 95]))
    if maximo - minimo < 0.3:
        print("Calibracao falhou: abaixe e levante mais o indicador. Aperte C.")
        return None
    print(f"Calibrado: abertura {minimo:.2f} = 0% e {maximo:.2f} = 100%")
    return minimo, maximo
# ------------------------------------------------------------------------

# HUD --------------------------------------------------------------------
def texto(tela, mensagem, posicao, cor=BRANCO, escala=0.5):
    # putText não aceita acentos, por isso as mensagens da tela usam ASCII.
    cv2.putText(tela, mensagem, posicao, cv2.FONT_HERSHEY_SIMPLEX, escala, cor, 1, cv2.LINE_AA)


def desenhar_hud(frame, mascara, mao, nivel, status, limiar, fps):
    """Aplica o brilho e desenha contorno, pontos-chave, máscara e barra."""
    tela = cv2.convertScaleAbs(frame, alpha=nivel / 100)  # a detecção usa o original
    x1, y1, x2, y2 = ROI
    cv2.rectangle(tela, (x1, y1), (x2, y2), BRANCO, 1)
    roi = tela[y1:y2, x1:x2]  # desenha em coordenadas do retângulo
    if mao is not None:
        cv2.drawContours(roi, [mao["contorno"]], -1, VERDE, 2)
        cv2.drawContours(roi, [mao["casca"]], -1, AZUL, 1)
        cv2.circle(roi, mao["centro"], int(mao["raio"]), imagem_cinza, 1)
        if mao["furo"] is not None:
            cv2.drawContours(roi, [mao["furo"]], -1, AMARELO, 2)
        if mao["pinca"] is not None:
            p1, p2, vale = mao["pinca"]
            cv2.line(roi, p1, p2, AMARELO, 2)
            for ponto, cor in ((p1, AMARELO), (p2, AMARELO), (vale, VERMELHO)):
                cv2.circle(roi, ponto, 6, cor, -1)

    # Barra de 0 a 100 %, máscara binária em miniatura e instruções
    cv2.rectangle(tela, (20, 20), (300, 44), (60, 60, 60), -1)
    cv2.rectangle(tela, (20, 20), (20 + round(2.8 * nivel), 44), VERDE, -1)
    texto(tela, f"Brilho: {nivel:.0f}%", (20, 70), escala=0.7)
    mini = cv2.resize(mascara, None, fx=0.4, fy=0.4, interpolation=cv2.INTER_NEAREST)
    h, w = mini.shape
    tela[90:90 + h, 20:20 + w] = cv2.cvtColor(mini, cv2.COLOR_GRAY2BGR)
    cv2.rectangle(tela, (20, 90), (20 + w, 90 + h), imagem_cinza, 1)
    texto(tela, f"Limiar {limiar} | {fps:.0f} FPS", (20, 112 + h))
    tela = cv2.copyMakeBorder(tela, 0, 50, 0, 0, cv2.BORDER_CONSTANT, value=(30, 30, 30))
    texto(tela, status, (12, 500), AMARELO)
    texto(tela, "ESPACO fundo | C calibrar | +/- limiar | Q sair", (12, 522), imagem_cinza)
    return tela
# ------------------------------------------------------------------------

def main():
    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        print("Nao foi possivel abrir a webcam.")
        return
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    x1, y1, x2, y2 = ROI
    fundo, limiar, quadros_fundo = None, 10, None  # lista só durante a captura
    calibracao, amostras, inicio_cal = None, [], None
    historico = deque(maxlen=5)  # mediana móvel: ignora medidas isoladas erradas
    nivel, fps, anterior = 100.0, 0.0, perf_counter()
    try:
        while True:
            ok, frame = camera.read()
            if not ok:
                print("A webcam parou de enviar imagens.")
                break
            frame = cv2.resize(cv2.flip(frame, 1), (640, 480))  # imagem espelhada
            imagem_processada = preprocessamento(frame[y1:y2, x1:x2]) # processa a cada quadro a região da mão
            mascara, mao = np.zeros((y2 - y1, x2 - x1), np.uint8), None
            agora = perf_counter()

            # CALIBRAÇÃO E ANÁLISE DO FUNDO ------------------------------
            if quadros_fundo is not None:               
                quadros_fundo.append(imagem_processada)
                if len(quadros_fundo) == QUADROS_FUNDO:
                    fundo = calibrar_fundo(quadros_fundo)
                    quadros_fundo = None # volta a ser None até a próxima calibração
            elif fundo is not None:                     
                mascara = binarizacao(imagem_processada, fundo, limiar)
                mao = analisar_mao(mascara)
            abertura = None if mao is None else mao["abertura"]
            # ------------------------------------------------------------

            # CALIBRAÇÃO DO GESTO ----------------------------------------
            if inicio_cal is not None:                  # calibração do gesto
                if abertura is not None:
                    amostras.append(abertura)
                if agora - inicio_cal >= TEMPO_CALIBRACAO:
                    calibracao = calibrar_gesto(amostras) or calibracao
                    inicio_cal = None
            elif calibracao and abertura is not None:   # controle do brilho
                historico.append(abertura)
                minimo, maximo = calibracao
                proporcao = (np.median(historico) - minimo) / (maximo - minimo)
                nivel = float(np.clip(100 * proporcao, 0, 100))

            if quadros_fundo is not None:
                status = "Capturando o fundo: mantenha o retangulo vazio."
            elif fundo is None:
                status = "Deixe o retangulo vazio e aperte ESPACO."
            elif inicio_cal is not None:
                restante = TEMPO_CALIBRACAO - (agora - inicio_cal)
                status = f"Calibrando: abaixe e levante o indicador ({restante:.1f} s)."
            elif abertura is None:
                status = "Mostre o polegar para o lado e o indicador para cima."
            elif calibracao is None:
                status = "Aperte C e abaixe e levante o indicador por 3 s."
            else:
                status = "Abaixe o indicador para escurecer e levante para clarear."

            fps = 0.9 * fps + 0.1 / max(agora - anterior, 1e-3)
            anterior = agora
            tela = desenhar_hud(frame, mascara, mao, nivel, status, limiar, fps)
            cv2.imshow(JANELA, tela)

            tecla = chr(cv2.waitKey(1) & 0xFF).lower()
            if tecla in ("q", "\x1b") or cv2.getWindowProperty(JANELA, cv2.WND_PROP_VISIBLE) < 1:
                break
            if tecla == " ":
                quadros_fundo = []
            elif tecla == "c" and fundo is not None and inicio_cal is None:
                amostras, inicio_cal = [], agora
            elif tecla in ("+", "="):
                limiar = min(limiar + 2, 100)
            elif tecla == "-":
                limiar = max(limiar - 2, 5)
    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
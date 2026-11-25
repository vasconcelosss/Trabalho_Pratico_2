import gc
import statistics
import time
import tracemalloc


def medir_tempo(funcao, *args, repeticoes=1, desabilitar_gc=False, **kwargs):
    tempos = []
    resultado = None
    for _ in range(max(1, repeticoes)):
        if desabilitar_gc:
            gc.collect()
            gc.disable()
        try:
            t0 = time.perf_counter()
            resultado = funcao(*args, **kwargs)
            tempos.append(time.perf_counter() - t0)
        finally:
            if desabilitar_gc:
                gc.enable()
    return resultado, statistics.median(tempos)


def medir_memoria_kb(funcao, *args, **kwargs):
    ja_ativo = tracemalloc.is_tracing()
    if not ja_ativo:
        tracemalloc.start()
    else:
        tracemalloc.reset_peak()
    try:
        resultado = funcao(*args, **kwargs)
        _, pico = tracemalloc.get_traced_memory()
    finally:
        if not ja_ativo:
            tracemalloc.stop()
    return resultado, pico / 1024.0


def medir(funcao, *args, repeticoes=5, desabilitar_gc=False, **kwargs):
    resultado, tempo_s = medir_tempo(
        funcao, *args, repeticoes=repeticoes, desabilitar_gc=desabilitar_gc, **kwargs
    )
    _, pico_kb = medir_memoria_kb(funcao, *args, **kwargs)
    return resultado, tempo_s, pico_kb
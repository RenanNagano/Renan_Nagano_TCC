import pandas as pd
import re

# 1. Carregar dataset original
df = pd.read_csv('(GPT)coleta01_lula_bc_master.csv')
df.columns = [c.strip() for c in df.columns]

def classificar_comentario(texto):
    t_clean = str(texto).strip()
    t = t_clean.lower()
    
    # Indicadores de Polaridade
    pos_terms = [
        'estamos juntos', 'parabéns', 'parabens', 'viva lula', 'lula la', 'lula lá', 'meu presidente', 
        'presidente lula', 'estamos orando', 'força lula', 'lula 2026', 'é isso aí', 'deus abençoe', 
        'gigante', 'nos representa', 'orgulho', 'boa lula', 'fala mesmo', 'coragem', 'lula tetra', '1313'
    ]
    
    neg_terms = [
        'lava jato', 'roubo', 'roubou', 'escandalo', 'escândalo', 'desgoverno', 'mentira', 'engano', 
        'fora lula', 'não fez nada', 'afundar', 'inss', 'sigilo', 'envolvido', 'vergonha', 'farsa', 
        'piada', 'mentiroso', 'impeachment', 'pior', 'caladinho', 'preso', 'cadeia', 'cara de pau'
    ]
    
    has_pos = any(p in t for p in pos_terms)
    has_neg = any(n in t for n in neg_terms) or (is_incivil == 1)
    
    if any(e in t_clean for e in ['😂', '😅', '🤡', '🤮', '💩']):
        if any(w in t for w in ['inocente', 'ué', 'quem', 'fala', 'lulinha', 'kkk', 'olha']):
            has_neg, has_pos = True, False
            
    pos, neg, neu = 0, 0, 0
    if has_pos and not has_neg:
        pos = 1
    elif has_neg and not has_pos:
        neg = 1
    elif has_pos and has_neg:
        pos = 1 if any(w in t for w in ['lula 2026', 'meu presidente', 'estamos juntos']) else 0
        neg = 1 - pos
    else:
        neu = 1
        
    return pos, neg, neu, is_civil, is_incivil

# Aplicar e salvar
resultados = [classificar_comentario(c) for c in df['comentario']]
df['POSITIVO'] = [r[0] for r in resultados]
df['NEGATIVO'] = [r[1] for r in resultados]
df['NEUTRO'] = [r[2] for r in resultados]
df['CIVIL'] = [r[3] for r in resultados]
df['INCIVIL'] = [r[4] for r in resultados]

df.to_csv('coleta01_lula_bc_master_classificado.csv', index=False, encoding='utf-8-sig')
print("Arquivo salvo com sucesso!")

# Etapa 1 - Analise Exploratoria de Dados (AED)

## Q1. Quantas linhas e colunas o dataset possui? E o esperado conforme a documentacao do UCI?
- O dataset possui 13.611 linhas e 17 colunas.
- Isso confere com o esperado no UCI: 16 features de entrada + 1 coluna alvo (Class).

## Q2. Ha valores ausentes? De que tipo sao as features?
- Nao ha valores ausentes apos o carregamento robusto e conversao numerica.
- As 16 features sao numericas continuas (int64 e float64).

## Q3. Existem classes com muito menos amostras? O que isso pode causar no treinamento? Como a acuracia pode nos enganar nesse caso?
- Sim. Ha desbalanceamento entre classes (maior/menor ~= 6,79).
- Exemplo: DERMASON (3546) versus BOMBAY (522).
- Isso pode levar o modelo a privilegiar classes majoritarias e errar mais nas minoritarias.
- A acuracia pode parecer alta mesmo com desempenho ruim em classes pequenas. Por isso, F1-macro e metricas por classe sao essenciais.

## Q4. Quantos registros duplicados foram encontrados? Qual seria o impacto de mante-los?
- Foram encontrados 68 registros duplicados.
- Manter duplicatas pode enviesar o treinamento, reforcar padroes repetidos e superestimar desempenho em validacao/teste.

## Q5. Quais features apresentam distribuicao assimetrica com cauda longa?
Usando |skew| > 1.0, as candidatas foram:
- Area
- ConvexArea
- ShapeFactor4
- Solidity
- MinorAxisLength
- EquivDiameter
- Perimeter
- MajorAxisLength
- Eccentricity

## Q6. A transformacao log alterou visivelmente a distribuicao? Como isso pode beneficiar o treinamento?
- Sim. O log1p reduziu caudas longas e diminuiu a assimetria em varias features.
- Isso tende a estabilizar gradientes e diminuir a influencia de valores extremos no treinamento da rede.

## Q7. Identifique pelo menos 3 pares de features com correlacao acima de 0.90.
Exemplos encontrados:
- Area x EquivDiameter: 1.000
- Area x ConvexArea: 0.9999
- ConvexArea x EquivDiameter: 0.9999
- Compactness x ShapeFactor3: 0.9987

## Q8. O que e multicolinearidade? Features muito correlacionadas trazem informacao nova ou sao redundantes? Features muito correlacionadas sao obrigadas a serem removidas?
- Multicolinearidade ocorre quando duas ou mais features sao altamente correlacionadas.
- Em geral, essas features trazem informacao redundante.
- Nao e obrigatorio remover todas. Em redes neurais, mesmo com correlacao alta, ainda e possivel obter bons resultados.
- A decisao depende de comparacao empirica (ex.: experimento com todas as features vs features selecionadas).

## Q9. Com base no heatmap e nos boxplots, quais features selecionar para treinar o modelo?
Selecao preliminar para a Etapa 2:
- Area
- MinorAxisLength
- Eccentricity
- Extent
- Solidity
- Roundness
- ShapeFactor1
- ShapeFactor3

## Q10. Quais features melhor separam as classes visualmente?
- Area e ShapeFactor1 mostraram boa separacao para algumas classes (especialmente BOMBAY e DERMASON em extremos).
- Roundness e Eccentricity tambem ajudam a separar grupos, embora com alguma sobreposicao.

## Q11. Quais classes apresentam maior sobreposicao? Voce espera que o modelo erre mais nessas classes?
- As maiores sobreposicoes visuais aparecem entre CALI, SIRA e HOROZ em algumas features.
- Tambem ha proximidade entre BARBUNYA e SIRA em certos intervalos.
- Sim, e esperado que o modelo apresente mais confusoes nesses pares/classes.

# Desafio de Estimativa de Fretes

## Sobre o Dado

- A base de cotações **"freight_costs.csv"** contém um subset de cotações com origem no estado do Mato Grosso para múltiplos destinos.
- A base de distâncias rodoviárias **"distances.csv"** contém um subset de distâncias com origem no estado do Mato Grosso para todos os demais municípios do Brasil.
- A coluna **ID_CITY_ORIGIN** representa o município de origem e a coluna **ID_CITY_DESTINATION** representa o município de destino da carga presentes nas bases de cotações e distâncias. Os códigos dos municípios seguem o padrão do IBGE.

## Output 1

Base de dados (formato CSV) histórica com as cotações de todos os municípios do Mato Grosso para os seguintes destinos: 1501303, 1506807, 3205309, 3548500, 4118204, 4207304, 4216206, 4315602.

Todos os municípios do Mato Grosso estão contidos na base de distâncias, isto é, são todos os valores únicos da coluna **ID_CITY_ORIGIN**.

## Output 2

Base de dados (formato CSV) com as cotações estimadas para as próximas 52 semanas para esses mesmos trajetos.

## Considerações Extras quanto à Avaliação do Desafio

Ambos os desafios buscam avaliar a capacidade de resolução de problemas pelo candidato.

- **Tarefa de Expansão das Cotações:**
  - Avalia a capacidade de construir pipelines/processos que sejam robustos e atualizáveis.
  - Observa boas práticas de programação.

- **Tarefa de Projeção:**
  - Avalia o processo de modelagem matemática e raciocínio lógico.
  - Não existe uma solução correta.
  - Avaliaremos principalmente o uso de ferramentas e técnicas de modelagem matemática utilizadas e pela coerência das hipóteses admitidas.

Essas tarefas têm como objetivo observar a metodologia do candidato, a abordagem analítica e a habilidade em lidar com dados complexos de forma eficiente e precisa.

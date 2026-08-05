

# Greek GPT

Este es un primer intento de crear un modelo de lenguaje MoE con únicamente texto en griego.
La implementación se realiza utilizando el framework Apple MLX.

## Datos

Los datos utilizados para el entrenamiento y la validación son la [Versión en griego de Wikipedia](https://huggingface.co/datasets/wikimedia/wikipedia/viewer/20231101.el), que consta de 227k enlaces URL de contenido de texto de Wikipedia.

## Modelos

Se desarrollaron dos modelos transformer:

- Uno con arquitectura MoE con 10,5M de parámetros.
- Un transformer Dense con 10,5M de parámetros.

Además, se crearon y entrenaron algunos NGrams sofisticados con fines de comparación.

- De 2-Gram a 5-Gram.

## Tokenizador

En cuanto al tokenizador, se utilizó el de GPT-2 de Hugging Face y se volvió a entrenar con nuestro conjunto de datos. El tamaño del vocabulario se configuró en 5000 tokens. No se dio un enfoque particular al entrenamiento del tokenizador, pero es una parte esencial de la modelización del lenguaje y es tan importante como el propio modelo.

## Resultados

Los resultados se pueden encontrar en el directorio `benchmarks/results`. El Transformer MoE no pareció ser superior al Dense, lo cual era de esperar, pero tampoco fue más rápido como ha mencionado todo el mundo.

<div align="center">

|         |  CE    | PPL    | Tiempo de inferencia (para 800 tokens) |
| :-:     | :---:  | :---:  | :---:                           |
| MoE     | 3.629  | 37.659 | ~47 segundos                     |
| Dense   | 3.616  | 37.184 | ~32 segundos                     |

</div>

La ventaja de velocidad de las arquitecturas MoE probablemente proviene de las técnicas de red aplicadas al entrenar un modelo en clústeres de GPU, en lugar de en una sola GPU.

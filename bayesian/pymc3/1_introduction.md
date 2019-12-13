---
layout: default
---
# Modelagem Bayesiana com PyMC3

> O PyMC3 permite escrever modelos usando uma sintaxe intuitiva para 
descrever um processo de geração de dados.

## Instalação (via pip)
```console
pip3 install pymc3
```

## Distribuições de Probabilidade no PyMC3
Para descrever uma função densidade de probabilidade no PyMC3 é simples,
como exemplo, a distribuição normal padrão é definida da seguinte forma:

```python
import pymc3 as pm

with pm.Model():

    x = pm.Normal('X', mu=0, sigma=1)
```

Dessa forma estamos definindo um variável (escalar) Normal como uma  *priori*,
ou seja, para especificar a variável `x` estamos instanciando um
método da classe `Normal`.

A variável requer ao menos um `name` como argumento de sua assinatura, 
e cada função de probabilidade tem seus devidos parâmetros próprios,
como no caso da Normal temos os parâmetros `mu ($\mu$)` e `sigma ($\sigma$)`.

No quadro anterior criamos uma variável aleatória escalar. Para criarmos uma 
variável aleatória vetorial precisamos definir a sua *forma* (`shape`):

```python
with pm.Model():
    y = pm.Beta('p', 1, 1, shape=(3, 3))
``` 

Acima acabamos de descrever 
$$ 
y \approx Beta(1,1) , y \in \R^{3} 
$$

No PyMC3 todas as distribuições de probabilidades são subclasses da Classe `Distribuition`, 
o qual por sua vez tem duas subclasses `Discrete` e `Continuous`, que são
variáveis definidas em **Theano** (o PyMC4 está sendo reconstruído utilizando o TensorFlow 
como Backend).


Todas as distribuições de probabilidade em PyMC3 (`pm.Distribution`) contém dois 
importantes métodos: `random()` e `log()`:


Esses métodos também são utilizados internamente pelo PyMC3 para realizar inferências.
O PyMC3 utiliza a *log-probabilidade*, no caso do `logp()`, para ajuste de modelos e 
o método `random()` é utilizado para fazer a amostragem da *posteriori*.

As assinaturas das funções são:

```python
x.randon(point=None, size=None)
```
```
x.logp(value)
```


## Utilizando uma distribuição sem um modelo
Os modelos que construímos nos tópicos anteriores estavam sendo sempre 
definidos dentro de um contexto chamado `Model`.
Caso utilizarmos uma função de probabilidade fora de um contexto, será retornado um erro:

```python
TypeError: No context on context stack
```

Isso ocorre por que todas as funções de probabilidade no PyMC3 foram feitas para serem
trabalhadas dentro de um contexto, no nosso caso `Model`.
Porém, cada uma `Distribution` tem um método `dist` que retorna um objeto mais
simplificado que nos permite trabalhar fora de um contexto. Por exemplo:

```python
import pymc3 as pm

y = pm.Binomial.dist(n=30, p=0.4)

print(y.logp(4).eval())
print(y.random(size=5))
```

Resultados:

```console
array(-6.72814841)
array([14, 15, 10, 11, 15])
```

##  Transformação automatizada do PyMC3
Para conseguir uma amostragem eficiente do MCMC, quaisquer variáveis 
contínuas que são restritas em sub-intervalo real $({\displaystyle \mathbb {R} } \mathbb{R}) $ 
são transformadas automaticamente para que seu suporte seja irrestrito. 


Definição de suporte:

$$
f: X \rightarrow Y
$$

$$
supp(f) := \{x \in X: f(x) \neq 0\}
$$


Essa transformação permite os algoritmos de amostragem trabalhar sem se 
preocupar com as restrições de limite.

Por exemplo, a distribuição gama é de valor positivo. Se definirmos um para um modelo:
```python
with pm.Model() as model:
    g = pm.Gamma('g', 1, 1)

print(model.vars)
print(model.deterministics)
```

Resultados:
```console
[g_log__]
[g]
```
Como o nome sugere, a variável `g` foi transformada em $log(g) $, 
e este é o espaço sobre o qual a amostragem ocorre.

Por padrão, as variáveis transformadas automaticamente são ignoradas 
ao imprimir e plotar a saída do modelo.

*Referências:* [Documentação](https://docs.pymc.io/Probability_Distributions.html)


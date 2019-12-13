---
layout: default
---
# PyMC3

> O PyMC3 permite que você escreva modelos usando uma sintaxe intuitiva para 
descrever um processo de geração de dados.

## Instalação(via pip)
``` console
name@host: ~$ pip3 install pymc3
```

## Distribuições de Probabilidades no PyMC3
Para descrever uma função densidade de probabilidade no PyMC3 é simples,
como exemplo, a distribuição normal padrão é definida da seguinte forma:

``` python3
import pymc3 as pm

with pm.Model():

    x = pm.Normal('X', mu=0, sigma=1)
```

Dessa forma estamos definindo um variável (escalar) Normal como *priori*,
ou seja, para especificar a variável `x` estamos instanciando um
método da class `Normal`.
A variável requer ao menos um `name` como argumento da sua assinatura, 
e cada função de probabilidade tem seus devidos parâmetros, como no caso da Normal
temos `mu` e `sigma`.

O exemplo anterior mostrou como criar uma variável aleatória escalar. Para criar uma 
variável aleatória vetorial precisamos definir a sua forma (`shape`):

```python3
with pm.Model():
    y = pm.Beta('p', 1, 1, shape=(3, 3))
``` 

Acimas acabamos de descrever $y \approx Beta(1,1) , y \in \R^{3} $.

No PyMC3 todas as distribuições de probabilidades são subclasses da Classe `Distribuition`, 
o qual por sua vez tem duas subclasses `Discrete` e `Continuous`, que são
variáveis definidas em **Theano** (o PyMC4 está sendo reconstruído utilizando o TensorFlow 
como Backend).


Todas as distribuições de probabilidade em PyMC3 (`pm.Distribution`) contém dois 
importantes métodos: `random()` e `log()`:


Os métodos acima também são utilizados internamente pelo PyMC3 para fazer a inferência 
utilizando o *log-porobabilidade*, no caso do `logp()`, para ajustar os modelos e 
o método `random()` é utilizado para fazer a amostragem da *posteriori*.

Suas assinaturas são as seguintes:

``` python3
x.randon(point=None, size=None)

x.logp(value)
```


## Utilizando uma distribuição sem um modelo
Os modelos que construímos nos tópicos anteriores estavam sendo definidos dentro de um contexto
chamado `Model`.
Caso utilizarmos uma função de probabilidade fora de um contexto, será lançado um erro:

``` python3
TypeError: No context on context stack
```

Isso ocorre por que todas as funções de probabilidade no PyMC3 foram feitas para se 
trabalhar dentro de um contexto.
Porém, cada uma `Distribution` tem um método `dist` retorna um objeto de distribuição mais
simplificado que nos permite trabalhar fora de um contexto.

``` python3
import pymc3 as pm

y = pm.Binomial.dist(n=30, p=0.4)

y.logp(4).eval()
y.random(size=5)
```

Resultados:

```console
array(-6.72814841)
array([14, 15, 10, 11, 15])
```

##  Transformação automatizada do PyMC3
Para conseguir uma amostragem eficiente do MCMC, quaisquer variáveis 
contínuas que são restritas em sub-intervalo real ($\R$) são transformadas 
automaticamente para que seu suporte seja irrestrito, ou seja: 


Definição de suporte:
$$ f: \X \rightarrow \Y  $$
$$ supp(f) = {x \in \X: f(x) <> 0} $$


Essa transformação permite os algoritmos de amostragem trabalhar sem se 
preocupar com as restrições de limite.

Por exemplo, a distribuição gama é de valor positivo. Se definirmos um para um modelo:
``` python3
with pm.Model() as model:
    g = pm.Gamma('g', 1, 1)

model.vars
model.deterministics
```

``` console
[g_log__]
[g]
```

## Escrevendo uma função de probabilidade personalizada

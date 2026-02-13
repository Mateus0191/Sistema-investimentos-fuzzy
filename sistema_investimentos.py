import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt

renda = ctrl.Antecedent(np.arange(0,20001,1),'renda_mensal')
risco = ctrl.Antecedent(np.arange(0,101,1),'tolerancia_risco')
carteira = ctrl.Consequent(np.arange(0,101,1),'tipo_de_carteira')

renda['baixa'] = fuzz.gaussmf(renda.universe,0,5000)
renda['media'] = fuzz.gaussmf(renda.universe,10000,4000)
renda['alta'] = fuzz.gaussmf(renda.universe,20000,5000)

risco['conservador'] = fuzz.gaussmf(risco.universe,0,25)
risco['moderado'] = fuzz.gaussmf(risco.universe,50,20)
risco['Agressivo'] = fuzz.gaussmf(risco.universe,100,25)

risco['conservador'] = fuzz.gaussmf(risco.universe, 0, 25)
risco['moderado'] = fuzz.gaussmf(risco.universe, 50, 20)
risco['Agressivo'] = fuzz.gaussmf(risco.universe, 100, 25)

carteira['conservadora'] = fuzz.trimf(carteira.universe, [0, 0, 50])
carteira['moderado'] = fuzz.trimf(carteira.universe, [20, 50, 80])
carteira['robusta'] = fuzz.trimf(carteira.universe, [50, 100, 100])


rule1 = ctrl.Rule(renda['baixa'] & risco['conservador'], carteira['conservadora'])
rule2 = ctrl.Rule(renda['media'] & risco['moderado'], carteira['moderado'])
rule3 = ctrl.Rule(renda['alta'] | risco['Agressivo'], carteira['robusta'])
rule4 = ctrl.Rule(renda['baixa'] & risco['moderado'], carteira['conservadora'])

investimento_ctrl = ctrl.ControlSystem([rule1,rule2,rule3,rule4])
recomendacao = ctrl.ControlSystemSimulation(investimento_ctrl)

recomendacao.input['renda_mensal'] = 5000
recomendacao.input['tolerancia_risco'] = 30

recomendacao.compute()

print(f"Score da Carteira: {recomendacao.output['tipo_de_carteira']:.2f}")
carteira.view(sim=recomendacao)
plt.show()
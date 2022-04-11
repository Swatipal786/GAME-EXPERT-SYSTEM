#!/usr/bin/env python
# coding: utf-8

# In[17]:


import numpy as np
import skfuzzy as fuzz
import matplotlib.pyplot as plt
import math 


# In[18]:


x_critic = np.arange(0, 11, 1)
x_userrating = np.arange(0, 11, 1)
x_globalsales = np.arange(0, 21, 1)
x_year  = np.arange(0, 66, 5)
x_result  = np.arange(0, 2, 1)

print("Critic Score: ", x_critic)
print("User Rating: ", x_userrating)
print("Global Sales: ", x_globalsales)
print("Year: ", x_year)
print("Result: ", x_result)


# In[19]:


# Generate fuzzy membership functions

# critic
critic_bad = fuzz.trimf(x_critic, abc=[0, 0, 6])
critic_regular = fuzz.trimf(x_critic, abc=[5, 6, 7])
critic_good = fuzz.trimf(x_critic, abc=[6, 10, 10])

# userrating
userrating_bad = fuzz.trimf(x_userrating, abc=[0, 0, 6])
userrating_regular = fuzz.trimf(x_userrating, abc=[4.5, 6, 7.5])
userrating_good = fuzz.trimf(x_userrating, abc=[6, 10, 10])

# global sales
globalsales_low = fuzz.trimf(x_globalsales, abc=[0, 0, 10])
globalsales_regular = fuzz.trimf(x_globalsales, abc=[0, 10, 20])
globalsales_high = fuzz.trimf(x_globalsales, abc=[10, 20, 20])

# year
year_old = fuzz.trimf(x_year, abc=[0, 0, 33])
year_modern = fuzz.trimf(x_year, abc=[33, 65, 65])

# result
result_recommended = fuzz.trimf(x_result, abc=[0, 0, 0.5])
result_notrecommended = fuzz.trimf(x_result, abc=[0.5, 1, 1])


# In[20]:


print('x_dia: ',x_critic)
print('dia_nr: ', critic_bad)
print('dia_medium: ',critic_regular)
print('dia_high: ',critic_good)


# In[21]:


# Visualize these universes and membership functions
fig, (ax0, ax1, ax2, ax3, ax4) = plt.subplots(nrows=5, figsize=(8, 9))

ax0.plot(x_critic, critic_bad, 'b', linewidth=1.5, label='Bad')
ax0.plot(x_critic, critic_regular, 'g', linewidth=1.5, label='Regular')
ax0.plot(x_critic, critic_good, 'r', linewidth=1.5, label='Good')
ax0.set_title('Critic Review')
ax0.legend()

ax1.plot(x_userrating, userrating_bad, 'b', linewidth=1.5, label='Bad')
ax1.plot(x_userrating, userrating_regular, 'g', linewidth=1.5, label='Regular')
ax1.plot(x_userrating, userrating_good, 'r', linewidth=1.5, label='Good')
ax1.set_title('User Rating')
ax1.legend()

ax2.plot(x_globalsales, globalsales_low, 'b', linewidth=1.5, label='Low')
ax2.plot(x_globalsales, globalsales_regular, 'g', linewidth=1.5, label='Regular')
ax2.plot(x_globalsales, globalsales_high, 'r', linewidth=1.5, label='Good')
ax2.set_title('Global Sales')
ax2.legend()

ax3.plot(x_year, year_old, 'b', linewidth=1.5, label='Old')
ax3.plot(x_year, year_modern, 'g', linewidth=1.5, label='Modern')
ax3.set_title('Year')
ax3.legend()

ax4.plot(x_result, result_recommended, 'b', linewidth=1.5, label='Recommended')
ax4.plot(x_result, result_notrecommended, 'g', linewidth=1.5, label='Not Recommended')
ax4.set_title('Prediction')
ax4.legend()

# Turn off top/right axes
for ax in (ax0, ax1, ax2, ax3, ax4):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()

plt.tight_layout()


# In[22]:


#Fuzzification and Defuzzification

def fuzzify(ip_critic, ip_userrating, ip_globalsales, ip_year):
  results = []
  critic_level_bad = fuzz.interp_membership(x_critic, critic_bad, ip_critic)
  critic_level_regular = fuzz.interp_membership(x_critic, critic_regular, ip_critic)
  critic_level_good = fuzz.interp_membership(x_critic, critic_good, ip_critic)

  
  userrating_level_bad = fuzz.interp_membership(x_userrating, userrating_bad, ip_userrating)
  userrating_level_regular = fuzz.interp_membership(x_userrating, userrating_regular, ip_userrating)
  userrating_level_good = fuzz.interp_membership(x_userrating, userrating_good, ip_userrating)

  globalsales_level_low = fuzz.interp_membership(x_globalsales, globalsales_low, ip_globalsales)
  globalsales_level_regular = fuzz.interp_membership(x_globalsales, globalsales_regular, ip_globalsales)
  globalsales_level_high = fuzz.interp_membership(x_globalsales, globalsales_high, ip_globalsales)

  year_level_old = fuzz.interp_membership(x_year, year_old, ip_year)
  year_level_modern = fuzz.interp_membership(x_year, year_modern, ip_year)

  #Rule-1
  cmp_var_1 = np.fmin(globalsales_level_low, year_level_old) 
  cmp_var_2 = np.fmin(critic_level_bad, userrating_level_bad)
  active_rule1 = np.fmin(cmp_var_1, cmp_var_2)
  result_notrecommended.round(2)
  result_act_recommended1 = np.fmin(active_rule1, result_notrecommended)
  print(result_act_recommended1,"************** 1")
  results.append(result_act_recommended1)
  #Rule-2 
  cmp_var_1 = np.fmin(globalsales_level_low, year_level_old) 
  cmp_var_2 = np.fmin(critic_level_bad, userrating_level_regular)
  active_rule2 = np.fmin(cmp_var_1, cmp_var_2)
  result_notrecommended.round(2)
  result_act_recommended2 = np.fmin(active_rule2, result_notrecommended)
  print(result_act_recommended2,"************** 2")
  results.append(result_act_recommended2)

  #Rule-3  
  cmp_var_1 = np.fmin(critic_level_bad, userrating_level_regular)
  cmp_var_2 = np.fmin(globalsales_level_low, year_level_modern)
  active_rule3 = np.fmin(cmp_var_1, cmp_var_2)
  result_notrecommended.round(2)
  result_act_recommended3 = np.fmin(active_rule3, result_notrecommended)
  print(result_act_recommended2,"************** 3")
  results.append(result_act_recommended3)

  #Rule-4  
  cmp_var_1 = np.fmin(critic_level_bad, userrating_level_regular)
  cmp_var_2 = np.fmin(globalsales_level_regular, year_level_modern)
  active_rule4 = np.fmin(cmp_var_1, cmp_var_2)
  result_recommended.round(2)
  result_act_recommended4 = np.fmin(active_rule4, result_recommended)
  print(result_act_recommended4,"************** 4")
  results.append(result_act_recommended4)

  #Rule-5  
  cmp_var_1 = np.fmin(critic_level_bad, userrating_level_regular)
  cmp_var_2 = np.fmin(globalsales_level_high, year_level_modern)
  active_rule5 = np.fmin(cmp_var_1, cmp_var_2)
  result_recommended.round(2)
  result_act_recommended5 = np.fmin(active_rule5, result_notrecommended)
  print(result_act_recommended5,"************** 5")
  results.append(result_act_recommended5)

  #Rule-6  
  cmp_var_1 = np.fmin(critic_level_bad, userrating_level_good)
  cmp_var_2 = np.fmin(globalsales_level_low, year_level_modern)
  active_rule6 = np.fmin(cmp_var_1, cmp_var_2)
  result_recommended.round(2)
  result_act_recommended6 = np.fmin(active_rule6, result_recommended)
  print(result_act_recommended6,"************** 6")
  results.append(result_act_recommended6)

  #Rule-7  
  cmp_var_1 = np.fmin(critic_level_bad, userrating_level_good)
  cmp_var_2 = np.fmin(globalsales_level_regular, year_level_old)
  active_rule7 = np.fmin(cmp_var_1, cmp_var_2)
  result_recommended.round(2)
  result_act_recommended7 = np.fmin(active_rule7, result_recommended)
  print(result_act_recommended7,"************** 7")
  results.append(result_act_recommended7)

  #Rule-8  
  cmp_var_1 = np.fmin(critic_level_good, userrating_level_good)
  cmp_var_2 = np.fmin(globalsales_level_high, year_level_old)
  active_rule8 = np.fmin(cmp_var_1, cmp_var_2)
  result_recommended.round(2)
  result_act_recommended8 = np.fmin(active_rule8, result_recommended)
  print(result_act_recommended8,"************** 8")
  results.append(result_act_recommended8)

  #Rule-9  
  cmp_var_1 = np.fmin(critic_level_regular, userrating_level_bad)
  cmp_var_2 = np.fmin(globalsales_level_low, year_level_old)
  active_rule9 = np.fmin(cmp_var_1, cmp_var_2)
  result_notrecommended.round(2)
  result_act_recommended9 = np.fmin(active_rule9, result_notrecommended)
  print(result_act_recommended9,"************** 9")
  results.append(result_act_recommended9)

  #Rule-10  
  cmp_var_1 = np.fmin(critic_level_regular, userrating_level_regular)
  cmp_var_2 = np.fmin(globalsales_level_low, year_level_old)
  active_rule10 = np.fmin(cmp_var_1, cmp_var_2)
  result_notrecommended.round(2)
  result_act_recommended10 = np.fmin(active_rule10, result_notrecommended)
  print(result_act_recommended10,"************** 10")
  results.append(result_act_recommended10)

  #Rule-11  
  cmp_var_1 = np.fmin(critic_level_regular, userrating_level_regular)
  cmp_var_2 = np.fmin(globalsales_level_low, year_level_modern)
  active_rule11 = np.fmin(cmp_var_1, cmp_var_2)
  result_notrecommended.round(2)
  result_act_recommended11 = np.fmin(active_rule11, result_notrecommended)
  print(result_act_recommended11,"************** 11")
  results.append(result_act_recommended11)

  #Rule-12  
  cmp_var_1 = np.fmin(critic_level_good, userrating_level_regular)
  cmp_var_2 = np.fmin(globalsales_level_high, year_level_old)
  active_rule12 = np.fmin(cmp_var_1, cmp_var_2)
  result_recommended.round(2)
  result_act_recommended12 = np.fmin(active_rule12, result_recommended)
  print(result_act_recommended12,"************** 12")
  results.append(result_act_recommended12)

  #Rule-13  
  cmp_var_1 = np.fmin(critic_level_good, userrating_level_bad)
  cmp_var_2 = np.fmin(globalsales_level_low, year_level_modern)
  active_rule13 = np.fmin(cmp_var_1, cmp_var_2)
  result_notrecommended.round(2)
  result_act_recommended13 = np.fmin(active_rule13, result_notrecommended)
  print(result_act_recommended13,"************** 13")
  results.append(result_act_recommended13)

  #Rule-14  
  cmp_var_1 = np.fmin(critic_level_good, userrating_level_regular)
  cmp_var_2 = np.fmin(globalsales_level_high, year_level_old)
  active_rule14 = np.fmin(cmp_var_1, cmp_var_2)
  result_recommended.round(2)
  result_act_recommended14 = np.fmin(active_rule14, result_recommended)
  print(result_act_recommended14,"************** 14")
  results.append(result_act_recommended14)

  #Rule-15  
  cmp_var_1 = np.fmin(critic_level_good, userrating_level_regular)
  cmp_var_2 = np.fmin(globalsales_level_low, year_level_modern)
  active_rule15 = np.fmin(cmp_var_1, cmp_var_2)
  result_recommended.round(2)
  result_act_recommended15 = np.fmin(active_rule15, result_recommended)
  print(result_act_recommended15,"************** 15")
  results.append(result_act_recommended15)

  #Rule-16  
  cmp_var_1 = np.fmin(critic_level_good, userrating_level_good)
  cmp_var_2 = np.fmin(globalsales_level_high, year_level_modern)
  active_rule16 = np.fmin(cmp_var_1, cmp_var_2)
  result_recommended.round(2)
  result_act_recommended16 = np.fmin(active_rule16, result_recommended)
  print(result_act_recommended15,"************** 16")
  results.append(result_act_recommended16)

  aggregated123 = np.fmax(result_act_recommended1, np.fmax(result_act_recommended2, result_act_recommended3))
  print(aggregated123)

  aggregated12345 = np.fmax(aggregated123, np.fmax(result_act_recommended4, result_act_recommended5))
  print(aggregated12345)

  aggregated1234567 = np.fmax(aggregated12345, np.fmax(result_act_recommended6, result_act_recommended7))
  print(aggregated1234567)

  aggregated123456789 = np.fmax(aggregated1234567, np.fmax(result_act_recommended8, result_act_recommended9))
  print(aggregated123456789)

  aggregated1234567891011 = np.fmax(aggregated123456789, np.fmax(result_act_recommended10, result_act_recommended11))
  print(aggregated1234567891011)

  aggregated12345678910111213 = np.fmax(aggregated1234567891011, np.fmax(result_act_recommended12, result_act_recommended13))
  print(aggregated12345678910111213)

  aggregated123456789101112131415 = np.fmax(aggregated12345678910111213, np.fmax(result_act_recommended14, result_act_recommended15))
  print(aggregated123456789101112131415)

  aggregated = np.fmax(aggregated123456789101112131415,result_act_recommended16)
  print(aggregated)


  # print(np.argmax(results))
  # aggregated = np.argmax(results)
  # print(x_result)
  # print(aggregated)


  #final_output = fuzz.defuzz(x_result, aggregated, 'centroid')
  final_output = fuzz.defuzz(x_result, aggregated, 'som')
  #final_output = fuzz.defuzz(x_result, aggregated, 'mom')
  #final_output = fuzz.defuzz(x_result, aggregated, 'lom')
 
  print(final_output)
    
  print(round(final_output))
  result = round(final_output)  
  if round(final_output) == 0:
    print("Game Recommended")
    return 'Game Recommended'
  else:
    print("Not Recommended")
    return 'Not Recommended'
  
    


# In[24]:


# Testing the given method 
fuzzify(9, 9, 20,50)

def testfile():
    return 'This is test file'


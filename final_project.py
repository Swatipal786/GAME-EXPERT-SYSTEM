


import numpy as np
import skfuzzy as fuzz
import matplotlib.pyplot as plt
import math 





x_critic = np.arange(0, 11, 1)
x_userrating = np.arange(0, 11, 1)
x_globalsales = np.arange(0, 21, 1)
x_year  = np.arange(0, 66, 5)
x_output = np.arange(0, 2, 1)

print("Critic Scores: ", x_critic)
print("Global Sales: ", x_globalsales)
print("Users Ratings: ", x_userrating)
print("Year: ", x_year)
print("Output: ", x_output)




# critic
bad_critic = fuzz.trimf(x_critic, abc=[0, 0, 6])
normal_critic = fuzz.trimf(x_critic, abc=[5, 6, 7])
good_critic = fuzz.trimf(x_critic, abc=[6, 10, 10])

# userrating
bad_userrating = fuzz.trimf(x_userrating, abc=[0, 0, 6])
normal_userrating = fuzz.trimf(x_userrating, abc=[4.5, 6, 7.5])
good_userrating = fuzz.trimf(x_userrating, abc=[6, 10, 10])

# global sales
low_globalsales = fuzz.trimf(x_globalsales, abc=[0, 0, 10])
normal_globalsales = fuzz.trimf(x_globalsales, abc=[0, 10, 20])
high_globalsales = fuzz.trimf(x_globalsales, abc=[10, 20, 20])

# year
old_year = fuzz.trimf(x_year, abc=[0, 0, 33])
new_year = fuzz.trimf(x_year, abc=[33, 65, 65])

# result
suggested_output = fuzz.trimf(x_output, abc=[0, 0, 0.5])
notsuggested_output = fuzz.trimf(x_output, abc=[0.5, 1, 1])





print('dia_x: ',x_critic)
print('nr_dia: ', bad_critic)
print('med_dia: ',normal_critic)
print('high_dia: ',good_critic)





# Visualize these universes and membership functions
fig, (ax0, ax1, ax2, ax3, ax4) = plt.subplots(nrows=5, figsize=(8, 9))

ax0.plot(x_critic, bad_critic, 'b', linewidth=1.5, label='Bad')
ax0.plot(x_critic, normal_critic, 'g', linewidth=1.5, label='Normal')
ax0.plot(x_critic, good_critic, 'r', linewidth=1.5, label='Good')
ax0.set_title('Critics Reviews')
ax0.legend()

ax1.plot(x_userrating, bad_userrating, 'b', linewidth=1.5, label='Bad')
ax1.plot(x_userrating, normal_userrating, 'g', linewidth=1.5, label='Normal')
ax1.plot(x_userrating, good_userrating, 'r', linewidth=1.5, label='Good')
ax1.set_title('Users Rating')
ax1.legend()

ax2.plot(x_globalsales, low_globalsales, 'b', linewidth=1.5, label='Low')
ax2.plot(x_globalsales, normal_globalsales, 'g', linewidth=1.5, label='Normal')
ax2.plot(x_globalsales, high_globalsales, 'r', linewidth=1.5, label='Good')
ax2.set_title('Global Sales')
ax2.legend()

ax3.plot(x_year, old_year, 'b', linewidth=1.5, label='Old')
ax3.plot(x_year, new_year, 'g', linewidth=1.5, label='New')
ax3.set_title('Year')
ax3.legend()

ax4.plot(x_output, suggested_output, 'b', linewidth=1.5, label='Suggested')
ax4.plot(x_output, notsuggested_output, 'g', linewidth=1.5, label='Not Suggested')
ax4.set_title('Recommendation')
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

def fuzzify(en_critic, en_globalsales, en_userrating, en_year):
  output = []
  critic_level_bad = fuzz.interp_membership(x_critic, bad_critic, en_critic)
  critic_level_Normal = fuzz.interp_membership(x_critic, normal_critic, en_critic)
  critic_level_good = fuzz.interp_membership(x_critic, good_critic, en_critic)

  
  userrating_level_bad = fuzz.interp_membership(x_userrating, bad_userrating, en_userrating)
  userrating_level_Normal = fuzz.interp_membership(x_userrating, normal_userrating, en_userrating)
  userrating_level_good = fuzz.interp_membership(x_userrating, good_userrating, en_userrating)

  globalsales_level_low = fuzz.interp_membership(x_globalsales, low_globalsales, en_globalsales)
  globalsales_level_Normal = fuzz.interp_membership(x_globalsales, normal_globalsales, en_globalsales)
  globalsales_level_high = fuzz.interp_membership(x_globalsales, high_globalsales, en_globalsales)

  year_level_old = fuzz.interp_membership(x_year, old_year, en_year)
  year_level_New = fuzz.interp_membership(x_year, new_year, en_year)

  #Rule-1
  comp_var1 = np.fmin(globalsales_level_low, year_level_old) 
  comp_var2 = np.fmin(critic_level_bad, userrating_level_bad)
  ActiveRule1 = np.fmin(comp_var1, comp_var2)
  notsuggested_output.round(2)
  output_action_suggested1 = np.fmin(ActiveRule1, notsuggested_output)
  print(output_action_suggested1,"************** 1")
  output.append(output_action_suggested1)
  #Rule-2 
  comp_var1 = np.fmin(globalsales_level_low, year_level_old) 
  comp_var2 = np.fmin(critic_level_bad, userrating_level_Normal)
  ActiveRule2 = np.fmin(comp_var1, comp_var2)
  notsuggested_output.round(2)
  output_action_suggested2 = np.fmin(ActiveRule2, notsuggested_output)
  print(output_action_suggested2,"************** 2")
  output.append(output_action_suggested2)

  #Rule-3  
  comp_var1 = np.fmin(critic_level_bad, userrating_level_Normal)
  comp_var2 = np.fmin(globalsales_level_low, year_level_New)
  ActiveRule3 = np.fmin(comp_var1, comp_var2)
  notsuggested_output.round(2)
  output_action_suggested3 = np.fmin(ActiveRule3, notsuggested_output)
  print(output_action_suggested2,"************** 3")
  output.append(output_action_suggested3)

  #Rule-4  
  comp_var1 = np.fmin(critic_level_bad, userrating_level_Normal)
  comp_var2 = np.fmin(globalsales_level_Normal, year_level_New)
  ActiveRule4 = np.fmin(comp_var1, comp_var2)
  suggested_output.round(2)
  output_action_suggested4 = np.fmin(ActiveRule4, suggested_output)
  print(output_action_suggested4,"************** 4")
  output.append(output_action_suggested4)

  #Rule-5  
  comp_var1 = np.fmin(critic_level_bad, userrating_level_Normal)
  comp_var2 = np.fmin(globalsales_level_high, year_level_New)
  ActiveRule5 = np.fmin(comp_var1, comp_var2)
  suggested_output.round(2)
  output_action_suggested5 = np.fmin(ActiveRule5, notsuggested_output)
  print(output_action_suggested5,"************** 5")
  output.append(output_action_suggested5)

  #Rule-6  
  comp_var1 = np.fmin(critic_level_bad, userrating_level_good)
  comp_var2 = np.fmin(globalsales_level_low, year_level_New)
  ActiveRule6 = np.fmin(comp_var1, comp_var2)
  suggested_output.round(2)
  output_action_suggested6 = np.fmin(ActiveRule6, suggested_output)
  print(output_action_suggested6,"************** 6")
  output.append(output_action_suggested6)

  #Rule-7  
  comp_var1 = np.fmin(critic_level_bad, userrating_level_good)
  comp_var2 = np.fmin(globalsales_level_Normal, year_level_old)
  ActiveRule7 = np.fmin(comp_var1, comp_var2)
  suggested_output.round(2)
  output_action_suggested7 = np.fmin(ActiveRule7, suggested_output)
  print(output_action_suggested7,"************** 7")
  output.append(output_action_suggested7)

  #Rule-8  
  comp_var1 = np.fmin(critic_level_good, userrating_level_good)
  comp_var2 = np.fmin(globalsales_level_high, year_level_old)
  ActiveRule8 = np.fmin(comp_var1, comp_var2)
  suggested_output.round(2)
  output_action_suggested8 = np.fmin(ActiveRule8, suggested_output)
  print(output_action_suggested8,"************** 8")
  output.append(output_action_suggested8)

  #Rule-9  
  comp_var1 = np.fmin(critic_level_Normal, userrating_level_bad)
  comp_var2 = np.fmin(globalsales_level_low, year_level_old)
  ActiveRule9 = np.fmin(comp_var1, comp_var2)
  notsuggested_output.round(2)
  output_action_suggested9 = np.fmin(ActiveRule9, notsuggested_output)
  print(output_action_suggested9,"************** 9")
  output.append(output_action_suggested9)

  #Rule-10  
  comp_var1 = np.fmin(critic_level_Normal, userrating_level_Normal)
  comp_var2 = np.fmin(globalsales_level_low, year_level_old)
  ActiveRule10 = np.fmin(comp_var1, comp_var2)
  notsuggested_output.round(2)
  output_action_suggested10 = np.fmin(ActiveRule10, notsuggested_output)
  print(output_action_suggested10,"************** 10")
  output.append(output_action_suggested10)

  #Rule-11  
  comp_var1 = np.fmin(critic_level_Normal, userrating_level_Normal)
  comp_var2 = np.fmin(globalsales_level_low, year_level_New)
  ActiveRule11 = np.fmin(comp_var1, comp_var2)
  notsuggested_output.round(2)
  output_action_suggested11 = np.fmin(ActiveRule11, notsuggested_output)
  print(output_action_suggested11,"************** 11")
  output.append(output_action_suggested11)

  #Rule-12  
  comp_var1 = np.fmin(critic_level_good, userrating_level_Normal)
  comp_var2 = np.fmin(globalsales_level_high, year_level_old)
  ActiveRule12 = np.fmin(comp_var1, comp_var2)
  suggested_output.round(2)
  output_action_suggested12 = np.fmin(ActiveRule12, suggested_output)
  print(output_action_suggested12,"************** 12")
  output.append(output_action_suggested12)

  #Rule-13  
  comp_var1 = np.fmin(critic_level_good, userrating_level_bad)
  comp_var2 = np.fmin(globalsales_level_low, year_level_New)
  ActiveRule13 = np.fmin(comp_var1, comp_var2)
  notsuggested_output.round(2)
  output_action_suggested13 = np.fmin(ActiveRule13, notsuggested_output)
  print(output_action_suggested13,"************** 13")
  output.append(output_action_suggested13)

  #Rule-14  
  comp_var1 = np.fmin(critic_level_good, userrating_level_Normal)
  comp_var2 = np.fmin(globalsales_level_high, year_level_old)
  ActiveRule14 = np.fmin(comp_var1, comp_var2)
  suggested_output.round(2)
  output_action_suggested14 = np.fmin(ActiveRule14, suggested_output)
  print(output_action_suggested14,"************** 14")
  output.append(output_action_suggested14)

  #Rule-15  
  comp_var1 = np.fmin(critic_level_good, userrating_level_Normal)
  comp_var2 = np.fmin(globalsales_level_low, year_level_New)
  ActiveRule15 = np.fmin(comp_var1, comp_var2)
  suggested_output.round(2)
  output_action_suggested15 = np.fmin(ActiveRule15, suggested_output)
  print(output_action_suggested15,"************** 15")
  output.append(output_action_suggested15)

  #Rule-16  
  comp_var1 = np.fmin(critic_level_good, userrating_level_good)
  comp_var2 = np.fmin(globalsales_level_high, year_level_New)
  ActiveRule16 = np.fmin(comp_var1, comp_var2)
  suggested_output.round(2)
  output_action_suggested16 = np.fmin(ActiveRule16, suggested_output)
  print(output_action_suggested15,"************** 16")
  output.append(output_action_suggested16)

  Aggregate_123 = np.fmax(output_action_suggested1, np.fmax(output_action_suggested2, output_action_suggested3))
  print(Aggregate_123)

  Aggregate_12345 = np.fmax(Aggregate_123, np.fmax(output_action_suggested4, output_action_suggested5))
  print(Aggregate_12345)

  Aggregate_1234567 = np.fmax(Aggregate_12345, np.fmax(output_action_suggested6, output_action_suggested7))
  print(Aggregate_1234567)

  Aggregate_123456789 = np.fmax(Aggregate_1234567, np.fmax(output_action_suggested8, output_action_suggested9))
  print(Aggregate_123456789)

  Aggregate_1234567891011 = np.fmax(Aggregate_123456789, np.fmax(output_action_suggested10, output_action_suggested11))
  print(Aggregate_1234567891011)

  Aggregate_12345678910111213 = np.fmax(Aggregate_1234567891011, np.fmax(output_action_suggested12, output_action_suggested13))
  print(Aggregate_12345678910111213)

  Aggregate_123456789101112131415 = np.fmax(Aggregate_12345678910111213, np.fmax(output_action_suggested14, output_action_suggested15))
  print(Aggregate_123456789101112131415)

  Aggregate = np.fmax(Aggregate_123456789101112131415,output_action_suggested16)
  print(Aggregate)


  # print(np.argmax(output))
  # Aggregate = np.argmax(output)
  # print(x_output)
  # print(Aggregate)


  #final_output = fuzz.defuzz(x_output, Aggregate, 'centroid')
  final_output = fuzz.defuzz(x_output, Aggregate, 'som')
  #final_output = fuzz.defuzz(x_output, Aggregate, 'mom')
  #final_output = fuzz.defuzz(x_output, Aggregate, 'lom')
 
  print(final_output)
    
  print(round(final_output))
  result = round(final_output)  
  if round(final_output) == 0:
    print("Game Suggested")
    return 'Game Suggested'
  else:
    print("Not Suggested")
    return 'Not Suggested'
  
    






fuzzify(9, 9, 20,50)

def testfile():
    return 'This is test file'


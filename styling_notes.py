# -*- coding: utf-8 -*-
"""
Styling notes
https://pandas.pydata.org/pandas-docs/stable/user_guide/style.html

@author: donbo
"""

import numpy as np
import pandas as pd

# row and column sums
df = pd.DataFrame({'a': [10,20],'b':[100,200],'c': ['a','b']})
df

# this does not change the df
df.append(df.sum().rename('Total'))
df


# this changes the df
df.loc['Column_Total']= df.sum(numeric_only=True, axis=0)
df.loc[:,'Row_Total'] = df.sum(numeric_only=True, axis=1)

print(df)

# pivot table
data = [('a',1,3.14),('b',3,2.72),('c',2,1.62),('d',9,1.41),('e',3,.58)]
df = pd.DataFrame(data, columns=('foo', 'bar', 'qux'))
df
df.pivot_table(index='foo',
               margins=True,
               margins_name='total',  # defaults to 'All'
               aggfunc=sum)


# does not work
df.pivot_table(margins=True,
               margins_name='total',  # defaults to 'All'
               aggfunc=sum)

total = df.apply(np.sum)
total['foo'] = 'tot'
df.append(pd.DataFrame(total.values, index=total.keys()).T, ignore_index=True)

# to html
df
df.to_html('temp.html')




                 a      b    c  Row_Total
0             10.0  100.0    a      110.0
1             20.0  200.0    b      220.0
Column_Total  30.0  300.0  NaN      330.0



df.head(10).style.format({"BasePay": "${:20,.0f}",
                          "OtherPay": "${:20,.0f}",
                          "TotalPay": "${:20,.0f}",
                          "TotalPayBenefits":"${:20,.0f}"})

temp = df.head(10).style.format({"BasePay": "${:20,.0f}",
                          "OtherPay": "${:20,.0f}",
                          "TotalPay": "${:20,.0f}",
                          "TotalPayBenefits":"${:20,.0f}"})
print(temp)

# lower case
df.head(10).style.format({"BasePay": "${:20,.0f}",
                          "OtherPay": "${:20,.0f}",
                          "TotalPay": "${:20,.0f}",
                          "TotalPayBenefits":"${:20,.0f}"})\
                 .format({"JobTitle": lambda x:x.lower(),
                          "EmployeeName": lambda x:x.lower()})

# hide index
df.head(10).style.format({"BasePay": "${:20,.0f}",
                          "OtherPay": "${:20,.0f}",
                          "TotalPay": "${:20,.0f}",
                          "TotalPayBenefits":"${:20,.0f}"})\
                 .format({"JobTitle": lambda x:x.lower(),
                          "EmployeeName": lambda x:x.lower()})\
                 .hide_index()

# conditional highlighting
df.head(10).style.format({"BasePay": "${:20,.0f}",
                          "OtherPay": "${:20,.0f}",
                          "TotalPay": "${:20,.0f}",
                          "TotalPayBenefits":"${:20,.0f}"})\
                 .format({"JobTitle": lambda x:x.lower(),
                          "EmployeeName": lambda x:x.lower()})\
                 .hide_index()\
                .highlight_max(color='lightgreen')\
                .highlight_min(color='#cd4f39')

# background
df.head(10).style.format({"BasePay": "${:20,.0f}",
                          "OtherPay": "${:20,.0f}",
                          "TotalPay": "${:20,.0f}",
                          "TotalPayBenefits":"${:20,.0f}"})\
                 .format({"JobTitle": lambda x:x.lower(),
                          "EmployeeName": lambda x:x.lower()})\
                 .hide_index()\
                .highlight_max(color='lightgreen')\
                .highlight_min(color='#cd4f39')

# properties for all cells
df.head(10).style.set_properties(**{'background-color': 'black',
                                    'color': 'lawngreen',
                                    'border-color': 'white'})

# selective formatting
df.head(10).style.format({"BasePay": "${:20,.0f}",
                          "OtherPay": "${:20,.0f}",
                          "TotalPay": "${:20,.0f}",
                          "TotalPayBenefits":"${:20,.0f}"})\
                 .format({"JobTitle": lambda x:x.lower(),
                          "EmployeeName": lambda x:x.lower()})\
                 .hide_index()\
                 .applymap(lambda x: f”color: {‘red’ if isinstance(x,str) else ‘black’}”)




tup = [('agi', 10, 0),
       ('agi ', 12, 1),
        ('agi', 13, 2),
        ('agi ', 14, 3),
        ('agi', 15, 4),
        ('agi ', 16, 5),
        ('wages', 17, 6)]
df2 = pd.DataFrame(tup, columns=['var', 'value', 'num'])
df2
df2['var'].unique()

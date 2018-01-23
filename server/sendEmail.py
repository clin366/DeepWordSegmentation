# coding: utf-8
'''
Description: 发送报警邮件
'''
import time
from email.mime.text import MIMEText
import smtplib

#===========================================================================
# 设置服务器，用户名、口令以及邮箱的后缀
#===========================================================================
mail_host="smtp.exmail.qq.com"
mail_user="robot@firedata.cc"
mail_pass="FireRobot001"
mail_postfix="Mail Alert"



def get_current_time():
    return time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))


# 发送邮件
def send_mail(sub,content):
    to_list = ['zengxiangneng@firedata.cc', 'shihengkun@firedata.cc', 'wangqi@firedata.cc']
    '''
    to_list:发给谁
    sub:主题
    content:内容
    '''
    me=mail_user+"<"+mail_user+"@"+mail_postfix+">"
    msg = MIMEText(content)
    msg['Subject'] = sub
    msg['From'] = me
    msg['To'] = ";".join(to_list)
    try:
        s = smtplib.SMTP()
        s.connect(mail_host)
        s.login(mail_user,mail_pass)
        s.sendmail(me, to_list, msg.as_string())
        s.close()
        return True
    except Exception, e:
        return False
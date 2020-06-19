import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('seaborn')
sns.set(font_scale=2)


train_df = pd.read_csv(r'input\train_ dataset\nCoV_100k_train.labled.csv',engine ='python')
test_df  = pd.read_csv(r'input\test_dataset\nCov_10k_test.csv',engine ='python')

# train_df = train_df[train_df['情感倾向'].isin(['0', '1', '-1'])]
# train_df['情感倾向'].value_counts().plot.bar()
# plt.title('sentiment(target)')
# plt.savefig("test.png",writer='pillow')
# plt.show()

#绘制四类图像并保持
def plot_data_1(train_df,test_df):
    train_df = train_df[train_df['情感倾向'].isin(['0','1','-1'])]
    train_df['情感倾向'].value_counts().plot.bar()
    plt.title('sentiment(target)')
    plt.savefig("1.png", writer='pillow')


def plot_data_2(train_df,test_df):
    train_df['time'] = pd.to_datetime('2020年' + train_df['微博发布时间'], format='%Y年%m月%d日 %H:%M', errors='ignore')

    train_df['month'] = train_df['time'].dt.month
    train_df['day'] = train_df['time'].dt.day
    train_df['dayfromzero'] = (train_df['month'] - 1) * 31 + train_df['day']

    fig, ax = plt.subplots(1, 2, figsize=(16, 8))
    sns.kdeplot(train_df.loc[train_df['情感倾向'] == '0', 'dayfromzero'], ax=ax[0], label='sent(0)')
    sns.kdeplot(train_df.loc[train_df['情感倾向'] == '1', 'dayfromzero'], ax=ax[0], label='sent(1)')
    sns.kdeplot(train_df.loc[train_df['情感倾向'] == '-1', 'dayfromzero'], ax=ax[0], label='sent(-1)')

    train_df.loc[train_df['情感倾向'] == '0', 'dayfromzero'].hist(ax=ax[1])
    train_df.loc[train_df['情感倾向'] == '1', 'dayfromzero'].hist(ax=ax[1])
    train_df.loc[train_df['情感倾向'] == '-1', 'dayfromzero'].hist(ax=ax[1])

    ax[1].legend(['sent(0)', 'sent(1)', 'sent(-1)'])
    plt.savefig("2.png", writer='pillow')



def plot_data_3(train_df,test_df):
    train_df['weibo_len'] = train_df['微博中文内容'].astype(str).apply(len)
    sns.kdeplot(train_df['weibo_len'])
    plt.title('weibo_len')
    plt.savefig("3.png", writer='pillow')


def plot_data_4(train_df,test_df):
    train_df['pic_len'] = train_df['微博图片'].apply(lambda x: len(eval(x)))
    train_df['pic_len'].value_counts().plot.bar()
    plt.title('pic_len(target)')
    plt.savefig("4.png", writer='pillow')

plot_data_1()
plot_data_2()
plot_data_3()
plot_data_4()
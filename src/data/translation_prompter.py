import random
class Prompter:
    english_prompts = [
        "Q: How do you say '<srctext>' in Chinese?\nA: You would say '<tgttext>'.",
        "I'm reading a book and came across this sentence '<srctext>'. I believe in Chinese it would be '<tgttext>'.",
        "Help needed! How would you translation '<srctext>' into '<srclang>'?\nI would translate it as '<tgttext>'",
        "During my English class, my teacher used the sentence '<srctext>'. It took me a while, but I finally translated it to '<tgttext>' in Chinese.",
        "In the movie I watched yesterday, a character said '<srctext>'. The subtitles translated it as '<tgttext>'.",
        "I came across the sentence '<srctext>' in an English novel. In the Chinese version of the book, it was translated as '<tgttext>'.",
        "In an English song, the lyrics go '<srctext>'. The Chinese version of the song translates it as '<tgttext>'.",
        "In an English speech, the speaker said '<srctext>'. The Chinese interpreter translated it as '<tgttext>'.",
        "I read an English news article which said '<srctext>'. The Chinese translation of the article had it as '<tgttext>'.",
        "The quote '<srctext>' from the English book translates to '<tgttext>' in the Chinese version.",
        "The English documentary mentioned '<srctext>'. The Chinese subtitles translated it as '<tgttext>'.",
        "The English speaker at the conference said '<srctext>'. The Chinese interpreter translated it as '<tgttext>'.",
        "[English]: '<srctext>'\n[Chinese]: '<tgttext>'",
        "Correct translation of the sentence '<srctext>' is '<tgttext>' in Chinese.",
        "User1: Can anyone help me translate this sentence into Chinese? '<srctext>'\nUser2: Sure, the translation would be '<tgttext>'.",
        "<srctext>\n<tgttext>"
    ]
    chinese_prompts = [
        "问题: <srctext> 用英语怎么说? 答: <tgttext>",
        "我在看书的时候碰到这么个句子:'<srctext>', 我觉得它的英文翻译是: '<tgttext>'",
        "帮帮我！'<srctext>' 的英文翻译是什么？\n 我觉得它应该翻译成'<tgttext>'.",
        "英语课上，我的老师说了一句话:'<srctext>',我想了很久，应该把它翻译成'<tgttext>'",
        "我昨天看的电影里面，有个人物说:'<srctext>', 字幕把它翻译成'<tgttext>'",
        "我在一本小说里碰到个句子:'<srctext>', 在这本书的英文译本里，它被翻译为'<tgttext>'",
        "在一首中文歌里, 有一句歌词'<srctext>', 这首歌的英文版本里把它翻译成'<tgttext>'",
        "我看到有个中文的新闻文章讲'<srctext>', 英文版本对应地把它翻译成了'<tgttext>'"
        "那个中文的纪录片提到'<srctext>', 英文的字幕把它翻译成'<tgttext>'",
        "在那个会议上，中文讲者说'<srctext>', 英文的口译人员把它翻译成'<tgttext>'",
        "中文： <srctext>\n 英文: <tgttext>",
        "句子'<srctext>'的英文翻译是:'<tgttext>'",
        "用户1: 谁能帮我把这个句子翻译成英文吗? '<srctext>'\n 用户2: 我来吧，它的翻译是'<tgttext>'",
        '<srctext>\n<tgttext>'
    ]

    @staticmethod
    def prompt_sentences(src_sents,tgt_sents):
        english_prompt = random.choice(Prompter.english_prompts)
        chinese_prompt = random.choice(Prompter.chinese_prompts)

        return [english_prompt.replace("<srctext>",src_sent.strip()).replace("<tgttext>",tgt_sent.strip()) for src_sent,tgt_sent in zip(src_sents,tgt_sents)], \
            [chinese_prompt.replace("<srctext>",tgt_sent.strip()).replace("<tgttext>",src_sent.strip()) for src_sent,tgt_sent in zip(src_sents,tgt_sents)]


        



import re

from config.config_logomotion import Animation_CONF, Gemini_CONF

class PopupAnimation:
    @staticmethod
    def get_prompt(prompt_lang):
        prompt_jp = """
画像1 は、アニメーションを生成したい広告の画像です。
また、下記のHTMLは上記の画像の広告をHTML形式で表しています。
```
[IMAGE_HTML]
```

上記の広告の画像に対してレイアウトとテキストの内容を考慮したうえで、レイヤーごとに分解された画像とテキストを画面に出現させるアニメーションのアイデアを生成してください。
動かしたいレイヤーのIDとその内容は下記のとおりです。
```
[LAYER_INFO]
```

また、実際にそのアニメーションを実現するアニメーションのスクリプトをanime.jsを使って作成してください。アニメーションの記述方法は以下の通りです。
- 1つのanime.timelineを使ってアニメーション.addで追加するような書き方です。
- 1つのレイヤーに対して1つのアニメーションを適用してください
- 同一グループ内のアニメーションは動きを連携させてください
- timelineの設定は`loop=false`としループしないように設定し、`autoplay=true`とし自動でアニメーションを開始するように設定します。

出力フォーマットは下記のとおりです。各項目を<タグ>で囲むように出力してください
```
## アニメーションのアイデア
<idea>
- **{アニメ－ショングループ１の名前}:** 
    1. {生成するアニメーションの説明}
    2. {生成するアニメーションの説明}
    ...

- **{アニメ－ショングループ２の名前}:** 
    1. {生成するアニメーションの説明}
    2. {生成するアニメーションの説明}
    ...

// アニメーショングループがあれば追加していく
...

</idea>

## anime.jsを活用したアニメーションのコード
<script>
const tl = anime.timeline({ loop: false, autoplay: true });

// アニメ－ショングループ１
tl.add({ 
    ... 
})
.add({
    ... 
});

// アニメ－ショングループ２
tl.add({
    ... 
})
.add({
    ... 
});

// アニメ－ショングループがあれば追加していく
...

</script>
```            
        """

        prompt_en = """
Given image 1 shows the thumbnail of an advertisement for which we want to generate animation.
Also, the HTML below represents the above advertisement image in HTML format.
```
[image_html]
```

Please generate an animation idea to make the layer decomposed image objects and text appear on screen, taking into account the layout and each content of the advertisement shown above.
The IDs and contents of the layers we want to animate are as follows:
```
[layergroup_info]
```

In addition, please generate an animation script using anime.js to implement these animations. The animation code should be written as follows:
- Use a single `anime.timeline` and add animations using `.add`
- Apply one animation per layer
- Coordinate movements of animations within the same group
- Configure the timeline settings with `loop=false` to prevent looping and `autoplay=true` to start the animation automatically.

The output format should be as follows. Please enclose each item with <tags>.
```

## Animation description
<desc>
- **{Animation group 1 name}:**
    1. {Description of animation}
    2. {Description of animation}
...

- **{Animation group 2 name}:**
    1. {Description of animation}
    2. {Description of animation}
...

// Add animation groups
...
</desc>

## Animation code
<script>
const tl = anime.timeline({ loop: false, autoplay: true });
// Animation group 1
tl.add({
    ...
})
.add({
    ...
});

// Animation group 2
tl.add({
    ...
})
.add({
    ...
});

// Add code for animation groups
...

</script>
```

        """

        return {
            "jp": prompt_jp,
            "en": prompt_en
        }[prompt_lang]

    @staticmethod
    def script_postprocess(script):
        match = [m.group() for m in re.finditer(r'anime.timeline\(\{(.*?)\}\)\;', script, re.DOTALL)]
        if match and Animation_CONF.options.loop is not None:
            timeline = match[-1] 
            loop_count = Animation_CONF.options.loop.count
            script = script.replace(timeline, "anime.timeline({loop: " + str(loop_count*2) + ", autoplay: true, direction: 'alternate', complete: function() {setTimeout(function() {document.getElementById('LOADING').style.display = \"block\";}, 1000);}});")
            return script
        
        elif match:
            timeline = match[-1]
            script = script.replace(timeline, "anime.timeline({loop: false, autoplay: true, complete: function() {setTimeout(function() {document.getElementById('LOADING').style.display = \"block\";}, 1000);}});")
            return script

        else:
            script = "/* Script has been deleted because the format is not correct. */ \n setTimeout(function() {document.getElementById('LOADING').style.display = \"block\";}, 5000);"  
            return script
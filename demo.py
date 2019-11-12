from bottle import route, run, template, request


@route('/piqa', methods=['GET', 'POST'])
def index():

    if request.method == "GET":
        # example =
        return template(

            """
            <form action="/piqa" method="post">
                Goal: <input name="goal" type="text" value="{{goal}}"/>
                Sol1: <input name="sol1" type="text" value="{{sol1}}"/>
                Sol2: <input name="sol2" type="text" value="{{sol2}}"/>
                <input value="Login" type="submit" />
            </form>
            """, goal=example["goal"], sol1=example["sol1"], sol2=example["sol2"])


run(host='localhost', port=8080)

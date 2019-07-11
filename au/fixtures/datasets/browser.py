import flask

app = flask.Flask(__name__)

from au.spark import Spark
spark = None
df = None

from flask import request
@app.route('/test', methods=['GET'])
def stream():
  # global spark
  # if not spark:
  #   spark = Spark.getOrCreate()
  # global df
  # if not df:
  #   from au.fixtures.datasets import argoverse as av
  #   df = av.Fixtures.label_df(spark, splits=('sample',))
  #   df = df.cache()
  #   df.show()

  from au.fixtures.datasets import argoverse as av

  uri = request.args.get('uri')
  from flask import stream_with_context
  @stream_with_context
  def generate(uri):
    
    if uri:
      uri = av.FrameURI.from_str(uri)
      frame = av.AVFrame(uri=uri)
      debug_img = frame.get_debug_image()

      import imageio
      from io import BytesIO
      out = BytesIO()
      imageio.imwrite(out, debug_img, format='jpg')
      from base64 import b64encode
      data = b64encode(out.getvalue()).decode('ascii')
      from urllib.parse import quote
      data_url = 'data:image/png;base64,{}'.format(quote(data))
      yield '<img src="%s"/ widht="800">' % data_url

    for r in range(1000):
      # df.registerTempTable('d')
      # yield str(spark.sql("select max(distance_meters), min(distance_meters) from d where camera = 'ring_front_center'  ").collect())
      yield str(uri) + '\n'
      yield str(r) + '\n'
      import time
      # time.sleep(1)
  return flask.Response(generate(uri), mimetype= 'text/html')

if __name__ == "__main__":
  app.run(debug=True)
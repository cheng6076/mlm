-- Mostly adapted from Wojciech Zaremba
local MaskedLoss, parent = torch.class('nn.MaskedLoss', 'nn.Module')

function MaskedLoss:__init()
   parent.__init(self)
end

function MaskedLoss:updateOutput(input)
  local input, target = unpack(input)
  local output = 0
  local acc = 0
  local normal = 0
  for i = 1, target:size(1) do
    if target[i] ~= 0 then
      if input[i]:max() == input[i][target[i]] then
        acc = acc + 1
      end
      normal = normal + 1
      output = output - input[i][target[i]]
    end
  end
  output = output / target:size(1)
  self.output = {output, acc, normal}
  return self.output
end

function MaskedLoss:updateGradInput(input)
  local input, target = unpack(input)
  self.gradInput:resizeAs(input)
  self.gradInput:zero()
  local z = -1 / target:size(1)
  local gradInput = self.gradInput
  for i=1,target:size(1) do
    if target[i] ~= 0 then
      gradInput[i][target[i]] = z
    end
  end
  return self.gradInput
end
